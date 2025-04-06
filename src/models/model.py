import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualConvUnit(nn.Module):
    """Residual Convolutional Unit"""
    def __init__(self, features, activation, bn):
        super().__init__()
        self.bn = bn
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)
        return self.skip_add.add(out, x)

class FeatureFusionBlock(nn.Module):
    """Feature Fusion Block"""
    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None):
        super().__init__()
        self.deconv = deconv
        self.align_corners = align_corners
        self.expand = expand
        out_features = features if not expand else features // 2
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True)
        self.rcu1 = ResidualConvUnit(features, activation, bn)
        self.rcu2 = ResidualConvUnit(features, activation, bn)
        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        output = xs[0]
        if len(xs) == 2:
            output = self.skip_add.add(output, self.rcu1(xs[1]))
        output = self.rcu2(output)
        modifier = {"scale_factor": 2} if (size is None and self.size is None) else {"size": size or self.size}
        output = F.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)
        return output

def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(features, nn.ReLU(False), deconv=False, bn=use_bn, expand=False, align_corners=True, size=size)

def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()
    # 각 레벨별 output 채널 (단순 구조 그대로)
    out_shape1, out_shape2, out_shape3 = out_shape, out_shape, out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape
    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8
    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    return scratch

# --- DPT Head 정의 ---
class DPTHead(nn.Module):
    def __init__(self, nclass, in_channels, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024]):
        super().__init__()
        self.nclass = nclass

        # 각 레벨별로 1x1 conv로 채널 맞추기
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels, out_ch, kernel_size=1, stride=1, padding=0)
            for out_ch in out_channels
        ])

        # 업샘플링 레이어: transposed conv 또는 identity 사용
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1)
        ])

        # DPT의 scratch (refinement) 모듈
        self.scratch = _make_scratch(out_channels, features, groups=1, expand=False)
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        # 출력 conv: 여기서는 depth map 예측 (nclass==1)
        self.scratch.output_conv1 = nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True)
        )
    def forward(self, features, patch_h, patch_w):
        outs = []
        # features는 각 레벨에 대한 tensor 리스트, cls token 무시하고 첫번째만 사용
        for i, x in enumerate(features): # 아 이게 4개네
            # x의 첫번째 원소만 사용 (batch, seq, dim) → (batch, dim, patch_h, patch_w)
            #print(x.shape)
            #x = x[0]
            x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[-1], patch_h, patch_w)
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            outs.append(x)
        layer1, layer2, layer3, layer4 = outs


        # scratch 모듈로 각 레벨 정제
        layer1_rn = self.scratch.layer1_rn(layer1)
        layer2_rn = self.scratch.layer2_rn(layer2)
        layer3_rn = self.scratch.layer3_rn(layer3)
        layer4_rn = self.scratch.layer4_rn(layer4)

        path4 = self.scratch.refinenet4(layer4_rn, size=layer3_rn.shape[2:])
        path3 = self.scratch.refinenet3(path4, layer3_rn, size=layer2_rn.shape[2:])
        path2 = self.scratch.refinenet2(path3, layer2_rn, size=layer1_rn.shape[2:])
        path1 = self.scratch.refinenet1(path2, layer1_rn)

        out = self.scratch.output_conv1(path1)
        # patch size에 맞춰 보간 → 최종 depth 해상도에 맞게 업샘플링
        out = F.interpolate(out, (patch_h * 14, patch_w * 14), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        return out

# --- DPT + DINOv2 Encoder ---
class DepthModel(nn.Module):
    def __init__(self, features=256, out_channels=[256, 512, 1024, 1024], use_bn=False, localhub=False, student=False):
        """
        encoder는 DINOv2 pretrained 모델
        DPT Head로 depth map 예측
        """
        super().__init__()
        # encoder: dino v2 vitb14 고정
        encoder = "vitb"
        if localhub:
            self.encoder = torch.hub.load(
                'torchhub/facebookresearch_dinov2_main',
                f'dinov2_{encoder}14',
                source='local',
                pretrained=True
            )
        else:
            self.encoder = torch.hub.load(
                'facebookresearch/dinov2',
                f'dinov2_{encoder}14'
            )

        # pretrained encoder의 첫 블록의 qkv in_features를 사용
        # 이거 실제 인풋이 아니라 채널만 맞춰주는 방식이야 !!
        in_channels = self.encoder.blocks[0].attn.qkv.in_features
        self.student = student

        # DPT Head: depth map 예측 (nclass=1)
        self.head = DPTHead(1, in_channels, features, use_bn, out_channels=out_channels)

    def forward(self, x):
        h, w = x.shape[-2:]
        # encoder로부터 중간 레벨 feature 4개 추출
        features = self.encoder.get_intermediate_layers(x, 4, return_class_token=False)
        patch_h, patch_w = h // 14, w // 14
        depth = self.head(features, patch_h, patch_w)
        # 최종 해상도에 맞게 보간
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = F.relu(depth)

        if self.student:
            """
            encoder result for feature alignment loss
            """
            #print(features.shape) -> 이거 결과가 튜플이라는데 ???
            #print(x.shape)

            ## feature alignment를 위한 encoder 값 추출
            # -> 근데 이거 솔직히 다 뽑을 필요 없고, feature alignment 하는거만 뽑아주면 되잖아
            # patch_features = final_features[:, :, :]    ## 여기서 지금 BxNxD 꼴인데 우리는 한 배치 안의 1st,2nd,3rd part 중에서 2nd 부분에 feature alignment 적용할거
            # 따라서 배치 크기를 3으로 나눠서, 그중 두번쨰 파트 부분만 가져오도록 하자
            # final_features = self.encoder(x,return_class_token=False) -> 공식 깃허브 확인 결과, 이거는 마지막 n개 layer를 반환하는 것 같음. 고로 그냥 -1 인덱싱
            final_features = features[-1] ## 이거 cls 없다고 판단했기 때문에 그냥 -1 한거임.
            valid_idx_range = x.shape[0]//3 # 어차피 그 전에 배치 3으로 나눠지는지 확인하니까, 굳이 여기선 안해도 괜춘 12 -> 4
            patch_features = final_features[valid_idx_range:valid_idx_range*2,:,:]   # B x N x D

            print(patch_features.shape) # 4,1024,768 나옴 -> 우리가 생각했던 차원 맞음 !!
            return depth.squeeze(1), patch_features

        return depth.squeeze(1)

# Depth_Anything

Depth_Anything is a deep learning project based on PyTorch that performs monocular depth estimation using a combination of DINOv2 and DPT Head.

---

To install the required packages, run:

```bash
pip install -r requirements.txt
```

---

To train the model, run:

```bash
python -m src.train (--teacher or --student)
```

---

To predict the model, run:

```bash
python -m src.pred (--teacher or --student)
```

# 📝 [MLCAS 2024 Corn Yield Prediction](https://eval.ai/web/challenges/challenge-page/2332/overview)

---

- Although this solution only stopped at top 10 in the LB, here is a quick note for feature selection that was used, which can be used as a reference for similar task:

    - Raw 6-band satellite data, resized to 24x24, normalized to [0,1]
    - Location encoding
    - Nitrogen level encoding
    - Passing days from the planting day

- I tested several models, but finally used a simple CNN taking one time-point data of a location as input, while using average of yield scores on all the time points as output for one location. The final model is [here](https://github.com/xmba15/corn_yield_prediction_mlcas/blob/master/src/models/corn_yield_model.py#L128)

- Several ideas that were tested but were not showing good results for my experiments:
    - cnn-lstm
    - contrastive learning to initialize weights for the models

## 🎛 Development environment

---

```bash
mamba env create --file environment.yml
mamba activate corn_yield
```

## :gem: References

---
- [MLCAS Corn Yield Prediction Using Satellite Data](https://eval.ai/web/challenges/challenge-page/2332/overview)

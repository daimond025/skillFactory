import pymc3 as pm
bi_model = pm.Model()

with pm.Model() as bi_model:
    pm.glm.GLM.from_formula('price_log ~ area + life + room + kitchen ', data, family=pm.glm.families.Binomial())
    trace = pm.sample(1000)


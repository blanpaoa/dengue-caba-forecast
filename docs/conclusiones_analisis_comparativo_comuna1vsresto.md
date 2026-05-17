# Conclusiones — Análisis Comparativo Comuna 1 vs Resto de CABA

**Sprint 3 — EDA Complementario**

---

## 1. Peso de la Comuna 1

La Comuna 1 concentra el **39.9% de todos los casos** del período (14,810 de 37,078) siendo una de las 15 comunas. Es un outlier espacial extremo — casi 6 veces más casos de lo esperado si la distribución fuera uniforme (6.7% por comuna).

---

## 2. Estacionalidad — consistente en toda CABA

El pico epidémico ocurre en la **SE 13 (~última semana de marzo)** tanto en la Comuna 1 como en el resto de las 14 comunas. La forma de la curva estacional es prácticamente idéntica en ambos grupos.

**Conclusión:** La estacionalidad es una característica de toda CABA, no de la Comuna 1. Las variables de temporada (`is_epidemic_season`, `semana_sin`, `semana_cos`) aplican globalmente sin necesidad de diferenciación por comuna.

---

## 3. Correlaciones clima-dengue — las conclusiones NO cambian

Todos los signos de correlación son consistentes en los tres grupos (CABA completo, solo Comuna 1, resto):

| Variable | Dirección | Consistencia |
|----------|-----------|-------------|
| Temperatura media | Positiva | ✓ igual en los 3 grupos |
| Precipitación | Positiva | ✓ igual en los 3 grupos |
| Humedad | Positiva | ✓ igual en los 3 grupos |

**Spearman detecta 2-3x más correlación que Pearson** para temperatura e índice de calor, confirmando que la relación es no lineal. Pearson la subestima porque asume una línea recta cuando en realidad hay un umbral térmico a partir del cual el mosquito se activa.

**Conclusión crítica:** La concentración de casos en la Comuna 1 **no distorsiona el análisis general**. Las relaciones clima-dengue identificadas en el EDA son válidas para toda CABA.

---

## 4. Desbalance de clases — la Comuna 1 es diferente

| Grupo | % semanas con 0 casos | Mediana semanas activas |
|-------|----------------------|------------------------|
| CABA completo | 69.0% | — |
| Solo Comuna 1 | **57.1%** | **18 casos** |
| Resto 14 comunas | 69.8% | 10 casos |

La Comuna 1 tiene **12 puntos porcentuales menos de semanas vacías** y el doble de intensidad típica cuando está activa. Esto indica transmisión más continua durante todo el año, posiblemente por su alta densidad de tránsito y acceso al sistema de salud.

---

## 5. Implicaciones para el modelado

| Decisión | Justificación |
|----------|--------------|
| Usar Spearman como métrica de correlación | Más robusta ante outliers y relaciones no lineales |
| Incluir `comuna_id` como feature categórica | La Comuna 1 tiene magnitud e intensidad propias |
| Reportar métricas separadas por comuna | El desbalance difiere entre grupos (57% vs 70% ceros) |
| Evaluar modelo específico para Comuna 1 | Su dinámica de transmisión es estructuralmente diferente |
| Features de estacionalidad aplican a toda CABA | Pico SE 13 consistente en todos los grupos |


# Unsupervised Learning: Segment, Detect, Propagate with Scikit-Learn

Chapter 8 of Hands-On Machine Learning as practitioners use it: choose the number of k-means clusters with inertia and silhouette scores, cluster the moons with DBSCAN and predict for new points through its core samples, fit Gaussian mixtures, pick the component count with BIC and turn density into an anomaly detector, then label only 50 representative digits and propagate their labels through the clusters to train a classifier that rivals one trained on far more labels - finishing with k-means color segmentation of an image.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** blobs_data
- [x] **2.** fit_kmeans
- [x] **3.** inertia_curve
- [x] **4.** silhouette_curve
- [x] **5.** fit_dbscan
- [x] **6.** dbscan_predict
- [x] **7.** fit_gmm
- [x] **8.** flag_anomalies
- [x] **9.** digits_data
- [x] **10.** baseline_50_random
- [x] **11.** representative_digits
- [x] **12.** train_on_representatives
- [x] **13.** propagate_and_train
- [x] **14.** synthetic_image
- [x] **15.** segment_colors
- [x] **16.** save_and_reload_clusterer
- [x] **17.** predict_digit_labels

## Results

```
inertia by k:   1:3,536  2:1,170  3:663  4:266  5:213  6:173  7:147  8:122
silhouette by k:2:0.589  3:0.569  4:0.685  5:0.657  6:0.604  7:0.556  8:0.561
silhouette picks k = 4; the data was generated with 5 blobs (three of them tightly packed, which is why 4 looks good too)
DBSCAN eps=0.05: 7 clusters, 77 noise points, 808 core samples
DBSCAN eps=0.2: 2 clusters, 0 noise points, 1000 core samples
new points assigned via core-sample KNN: [1, 0, 1, 0]
BIC by components: 2:5,657  3:5,197  4:3,964  5:3,470  6:3,512  7:3,554  -> 5 components
anomalies at 4% contamination: 80 of 2000 points flagged by low density

digits, 50-label budget:
  50 random labels             -> test accuracy 0.804
  50 representative labels     -> test accuracy 0.900
  propagated to 291 points (96.9% of them correctly) -> test accuracy 0.900
  every label (1347)        -> test accuracy 0.962

color segmentation: 49 colors -> 4
served on 8 raw images: [0, 1, 8, 3, 4, 3, 6, 7] (truth [0, 1, 2, 3, 4, 5, 6, 7])
```

# Retinal OCT Image Dataset

## About Dataset

Retinal optical coherence tomography (OCT) is an imaging technique used to capture high-resolution cross-sections of the retinas of living patients. Approximately 30 million OCT scans are performed each year, and the analysis and interpretation of these images take up a significant amount of time (Swanson and Fujimoto, 2017).

(A) (Far left) choroidal neovascularization (CNV) with neovascular membrane (white arrowheads) and associated subretinal fluid (arrows). (Middle left) Diabetic macular edema (DME) with retinal-thickening-associated intraretinal fluid (arrows). (Middle right) Multiple drusen (arrowheads) present in early AMD. (Far right) Normal retina with preserved foveal contour and absence of any retinal fluid/edema.

## Content

The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (NORMAL, CNV, DME, DRUSEN). There are 84,495 X-Ray images (JPEG) and 4 categories (NORMAL, CNV, DME, DRUSEN).

Images are labeled as (disease)-(randomized patient ID)-(image number by this patient) and split into 4 directories: CNV, DME, DRUSEN, and NORMAL.

Before training, each image went through a tiered grading system consisting of multiple layers of trained graders of increasing expertise for verification and correction of image labels. Each image imported into the database started with a label matching the most recent diagnosis of the patient. The dataset selection and stratification process is displayed in a CONSORT-style diagram.

For additional information, see [the original publication](http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5).

## Acknowledgements

- **Data**: [Mendeley Dataset](https://data.mendeley.com/datasets/rscbjbr9sj/2), [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/kermany2018)
- **Citation**: [Original Paper](http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5)

---

Feel free to explore and utilize this valuable dataset for your research and projects!

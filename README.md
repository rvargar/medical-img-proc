# medical-img-proc

WIP - plans and ToDos:
 - Cone beam simulator
 - Image classification for malignant cases
# Downloading sample data (e.g.: Lung HR-CT)
1. Downloading sample <b>Chest HR CT</b> data e.g. from the following source from cancer imaging archive:
https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/ [1] <br>
For downloading from he Cancer Imaging Archive, you also need data retriever tool:
https://wiki.cancerimagingarchive.net/display/NBIA/NBIA+Data+Retriever+Command-Line+Interface+Guide
2. you can download CT collection with the following command e.g.: <br>
`/opt/nbia-data-retriever/bin/nbia-data-retriever --cli Lung-PET-CT-Dx-NBIA-Manifest-122220.tcia -d ./data/`

[1] Li, P., Wang, S., Li, T., Lu, J., HuangFu, Y., & Wang, D. (2020). A Large-Scale CT and PET/CT Dataset for Lung Cancer Diagnosis (Lung-PET-CT-Dx) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.2020.NNC2-0461
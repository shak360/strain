# Strain Analysis

This project documents the process to obtain deep learning derived LV global longitudinal strain from an apical-4-chamber echocardiogram video.

## Setup
1. Get the LV segmentation model (deeplabv3_resnet50_random.pt) weights from EchoNet-Dynamic, or train your own LV segmention model using videos from https://github.com/echonet/dynamic
2. place all the strain videos as .avi files in a single folder

## Processing

You may now run the code using the examples/estimate_strain.ipynb notebook.

## Output
The four output locations are arguments to estimate_strain, and the strain itself is returned.
1. segmentation_dir is the folder that contains the videos of the segmentation. If something went wrong, check here, as echonet may not be processing the videos correctly. This is the only one of the four directories the code creates by itself.
2. strain_dir is the folder that contains the outline of the strain overlayed on the echocardiogram. Create it before running.
3. plot_dir is the folder that contains plots of the measurement of length in each frame. Create it before running.
4. excel_dir is the folder that contains a csv per video with the measurement of length in each frame. Create it before running.
5. estimate_strain returns the per-beat estimates (ratios L_ES / L_ED) for the video rather than writing a summary csv; examples/estimate_strain.ipynb collects them into its own csv. These estimates can be converted to traditional measurements of strain using (estimate*100) - 100, which is signed: shortening gives a negative strain. (This matches the conversion in examples/estimate_strain.ipynb; the previous form of this line, 100*(1-estimate), gave the same magnitude with the opposite sign.)

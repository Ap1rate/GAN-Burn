# Burn Wound ROI Detection and GAN-based Pseudo-Thermal Map Synthesis

This repository contains the implementation of our pipeline for **burn wound detection and pseudo-thermal image generation**, developed for **academic research and paper submission**.  
The project integrates **YOLOv11** for automatic Region of Interest (ROI) detection and a **GAN model (Swin-Unet Generator + PatchGAN Discriminator)** for generating pseudo-thermal representations, which are used to approximate burn depth estimation.

---

# GAN

<img width="2560" height="1341" alt="Figure_1" src="https://github.com/user-attachments/assets/94384b33-3273-4a5e-87ab-54d6dd68d702" />

Our GAN framework is based on **pix2pix**, extended with a **Swin-Unet backbone** for enhanced feature extraction and contextual understanding.  
To stabilize training and encourage structural consistency, we introduce a **Gradient Consistency Loss** in addition to the standard L1 and adversarial objectives.  
This design allows the generator to synthesize heat-map-like pseudo-thermal outputs from the refined burn wound ROIs.

![2](https://github.com/user-attachments/assets/ca3f77aa-ccdd-46ee-bd4d-b79dcd2f6ad7)

---

The workflow begins with **YOLOv11 detection** of burn wound regions, followed by manual refinement of the cropped ROI for ground truth generation.  
The generator then learns to translate the cropped ROI into pseudo-thermal maps that align with expert-verified burn severity references.  
This combination provides a feasible pipeline for **non-contact burn assessment** using only smartphone-captured images.

![1](https://github.com/user-attachments/assets/2eb59c4e-b611-4987-9c06-8f8c46412ca9)

---

## Methodology Overview

- **YOLOv11 Model**  
  Detects bounding boxes for burn areas from raw clinical or smartphone-captured images.  

- **ROI Refinement**  
  Experts adjust and validate the wound boundary, generating high-quality inputs for GAN training.  

- **GAN (Swin-Unet + PatchGAN)**  
  - *Generator*: Swin-Unet with multi-scale feature extraction.  
  - *Discriminator*: PatchGAN for local texture realism.  
  - *Losses*: Adversarial loss + L1 loss + Gradient Consistency loss.  

- **Outputs**  
  Pseudo-thermal burn maps that approximate tissue damage severity.

<img width="3179" height="1194" alt="图片X" src="https://github.com/user-attachments/assets/4a715489-4e74-46bc-bd0d-218019a43aa4" />

---

## Key Features
- 🔍 **Automated Detection**: YOLOv11 for robust burn ROI localization.  
- 🧠 **Deep Generative Model**: Swin-Unet Generator improves contextual depth estimation.  
- 🎯 **Custom Loss**: Gradient Consistency Loss enforces edge-aware thermal map generation.  
- 📱 **Real-World Ready**: Pipeline validated on smartphone-captured burn wound images.  

<img width="1412" height="762" alt="图片9" src="https://github.com/user-attachments/assets/d4f1370a-df39-4c6c-9085-521716c5bb0b" />

---

## Contribution
This project is part of an **academic study for journal submission**, aiming to provide an **AI-assisted approach for non-invasive burn depth estimation**.  
The code and methodology are shared here to ensure **reproducibility and transparency** for the research community.  

<img width="2127" height="1176" alt="图片8" src="https://github.com/user-attachments/assets/533609ec-7ece-45a8-86af-6abbbe9a7c54" />

---

## Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/YourName/GAN-Burn.git
   cd GAN-Burn
   
   ```bash
python train.py --dataroot ./datasets/burn_dataset --name burn_gan --model pix2pix --netG swinunet

   ```bash
python test.py --dataroot ./datasets/burn_dataset --name burn_gan --model pix2pix

License

This repository is released under an Academic Research License.
Usage is restricted to non-commercial and research purposes.

<img width="908" height="514" alt="图片6" src="https://github.com/user-attachments/assets/628762ed-61ad-4b81-bd4d-f44a5e4d2085" />
Citation

If you use this codebase in your research, please cite our work once the paper is published.
We will update this section with DOI and citation details after acceptance.

<img width="1810" height="1118" alt="图片4" src="https://github.com/user-attachments/assets/16a22a74-5e5f-49a0-96f5-acf8db416b84" />
<img width="1299" height="972" alt="图片3" src="https://github.com/user-attachments/assets/33500fd4-ca64-47a4-9081-460dddea226c" /> ```


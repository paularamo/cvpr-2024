
<p align="center">
  <img src="https://github.com/openvinotoolkit/openvino_notebooks/assets/10940214/55709b5a-684e-4bf7-b7f4-2fe10638c7fa" width=200/>
  <img src="https://user-images.githubusercontent.com/10940214/165389618-63e6b369-76cd-4880-9582-360c58c8675d.png" width=200/>
</p>

# Edge-Optimized Deep Learning: Harnessing Generative AI and Computer Vision with Open-Source Libraries.

## Organizers:

[Samet Akcay](https://www.linkedin.com/in/sametakcay/), [Paula Ramos](https://www.linkedin.com/in/paula-ramos-41097319/), [Ria Cheruvu](https://www.linkedin.com/in/ria-cheruvu-54348a173/), [Alexander Kozlov](https://www.linkedin.com/in/alexander-kozlov-8abb20b2/), [Zhen (Fiona) Zhao](https://www.linkedin.com/in/zhen-fiona-zhao-45b818a9/), [Zhuo Wu](https://www.linkedin.com/in/wuzhuo/), [Raymond Lo](https://www.linkedin.com/in/raymondlo84/), & [Yury Gorbachev](https://www.linkedin.com/in/yurygorbachev/)


## Overview:

This tutorial aims to guide researchers and practitioners in navigating the complex deep learning (DL) landscape, focusing on data management, training methodologies, optimization strategies, and deployment techniques. It highlights open-source libraries like the OpenVINO toolkit, OpenVINO Training eXtensions (OTX), and Neural Network Compression Frameworks (NNCF) in streamlining DL development. The tutorial covers how OTX 2.0 [1] simplifies the DL ecosystem (Computer Vision) by integrating various frameworks and ensuring a consistent experience across different platforms (MMLab [2], Lightning [3], or Anomalib [4]). It also demonstrates how to  fine-tune generative AI models, specifically Stable Diffusion SD with LoRA, and the benefits of customized models in reducing latency and enhancing efficiency. The tutorial explores fine-tuning visual prompting tasks, including Segment Anything Model (SAM). It explains how to fine-tune a SD model with custom data using multiple acceleration methods [5, 6], and how to deploy the fine-tuned model using OpenVINO Transformation Passes API [9]. Lastly, the tutorial focuses on model optimization capabilities for the inference phase, with the OpenVINO toolkit and OTX library integrating with NNCF [10] to refine neural networks and improve inference speed, especially on edge devices with limited resources. The tutorial includes demos showcasing how OpenVINO runtime API enables real-time inference on various devices.


## Prework:
We will share this section in April/2024.

## Outline

1.	OpenVINO, OpenVINO Training eXtensions (OTX) and NNCF. Fundamentals
2.	Module 1: Data management, training, and fine-tuning downstream Computer Vision tasks.
3.	Module 2: Optimize and run Gen AI pipelines on your laptop. SD with LoRA weights. 
4.	Module 3: Optimization with NNCF for Computer Vision and Gen AI (Multimodal).
5.	Module 4: Evaluate and deploy your solution as an edge-computing system. Multiple Computer Vision tasks and Gen AI pipelines on a wide range of HW.

## Slides
We will share this section in June/2024.

## References 
[1] Intel Corporation, "OpenVINO™ Training Extensions", [Online]. Available: https://github.com/openvinotoolkit/training_extensions. Intel Corporation, 2023. [Accessed 27 November 2023]. 

[2] OpenMMLab, [Online]. Available: https://github.com/open-mmlab.  [Accessed 27 November 2023]. 

[3] Lightning.ai, [Online]. Available: https://lightning.ai/. [Accessed 27 November 2023].

[4] Intel Corporation, "Anomalib”, [Online]. Available: https://github.com/openvinotoolkit/anomalib/tree/main. Intel Corporation, 2023. [Accessed 27 November 2023].

[5] Q. Fu, et al., "Deep Learning Models on CPUs: A Methodology for Efficient Training," arXiv preprint arXiv:2206.10034, 2022. 

[6] Intel Corporation, “Remote Tensor API of GPU Plugin,” Intel corporation, 2023. Available online: OpenVINO Documentation [Accessed 5 November 2023].

[7] Intel Corporation, "OpenVINO Stable Diffuison (with LoRA) C++ pipeline," Intel Corporation, 2023. Available online: OpenVINO Documentation. [Accessed 5 November 2023].

[8] S.Luo, et al., " LCM-LoRA: A Universal Stable-Diffusion Acceleration Module," arXiv:2311.05556 Available online: arXiv [Accessed 4 December 2023].

[9] Zhen Zhao and Kunda Xu, "Enable LoRA Weights with Stable Diffusion Controlnet Pipeline," Intel Corporation, 7 Aug. 2023. Available online: Intel Community Blog. [Accessed 5 November 2023].

[10] Intel Corporation, "Neural Network Compression Framework (NNCF)", [Online]. Available: https://github.com/openvinotoolkit/nncf. Intel Corporation, 2023. [Accessed 27 November 2023].





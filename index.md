
<p align="center">
  <img src="https://github.com/openvinotoolkit/openvino_notebooks/assets/10940214/55709b5a-684e-4bf7-b7f4-2fe10638c7fa" width=200/>
  <img src="https://user-images.githubusercontent.com/10940214/165389618-63e6b369-76cd-4880-9582-360c58c8675d.png" width=200/>
</p>

# Edge-Optimized Deep Learning: Harnessing Generative AI and Computer Vision with Open-Source Libraries.

## Organizers:

[Samet Akcay](https://www.linkedin.com/in/sametakcay/), [Paula Ramos](https://www.linkedin.com/in/paula-ramos-41097319/), [Ria Cheruvu](https://www.linkedin.com/in/ria-cheruvu-54348a173/), [Alexander Kozlov](https://www.linkedin.com/in/alexander-kozlov-8abb20b2/), [Zhen (Fiona) Zhao](https://www.linkedin.com/in/zhen-fiona-zhao-45b818a9/), [Zhuo Wu](https://www.linkedin.com/in/wuzhuo/), [Raymond Lo](https://www.linkedin.com/in/raymondlo84/), & [Yury Gorbachev](https://www.linkedin.com/in/yurygorbachev/)


## Overview:

This tutorial addresses the challenge of navigating the increasingly complex deep learning (DL) landscape, characterized by many frameworks with specialized functionalities. It aims to equip researchers and practitioners with the necessary skills to develop efficient and accessible DL models for diverse applications. This tutorial encompasses critical aspects of the DL pipeline, including robust data management, diverse training methodologies, optimization strategies, and efficient deployment techniques. Emphasis is placed on the utility of open-source libraries, such as, OpenVINO toolkit, OpenVINO Training eXtensions (OTX), and Neural Network Compression Frameworks (NNCF), in streamlining the DL development process. Through hands-on experiences with OpenVINO, OTX, and NNCF, participants will gain proficiency in managing data effectively, utilizing various training methods, and implementing optimizations across the AI lifecycle including computer vision pipelines and Generative AI (Gen AI). Furthermore, the tutorial dives into the concept of fine-tuning generative AI models, specifically Stable Diffusion SD with LoRA, adaptors for edge computing environments. This section highlights the advantages of customized models in reducing latency and enhancing efficiency. Ultimately, this comprehensive tutorial provides a valuable learning experience, equipping participants with the knowledge and skills necessary to navigate the complexities of modern DL and achieve success in their respective fields.
Tutorial attendees will evidence how OTX 2.0 [1] streamlines the complex deep learning ecosystem by providing a unified API and powerful CLI that integrate various frameworks, simplifying the process for researchers and developers. It ensures a consistent experience across different platforms (MMLab [2], Lightning [3], or Anomalib [4]) focusing on training, inference, and end-to-end optimization for edge deployment. Initially, the session will focus on fine-tuning downstream Computer Vision tasks such as detection and segmentation, covering techniques in supervised and semi/self-supervised learning. Additionally, the tutorial will explore the fine-tuning visual prompting tasks, including Segment Anything Model (SAM), showcasing the versatility of OTX 2.0 in addressing a wide range of computer vision challenges.
Building on the foundation of Computer Vision tasks, the tutorial will transition to the fine-tuning of Gen AI models.  This tutorial will explain how to fine-tune a SD model with custom data using multiple acceleration methods [5, 6], and how to deploy the fine-tuned model by inserting subgraph files with multiple LoRA weights into a single SD model. For the last one, we will use only the SD model graph once [7], instead of looping the SD model graph multiple times for multiple LoRA weights as in [8], so there are no extra execution costs in the forwarding operation of LoRA weights and shortened model compiling time could then be achieved, through using OpenVINO Transformation Passes API [9]. As a result, we can get an image with multiple features that represented in LoRA in just one inference.
Finally, the tutorial will focus on the model optimization capabilities for the inference phase of the AI lifecycle. OpenVINO toolkit and OTX library enhance model optimization by integrating with the Neural Network Compression Framework (NNCF) [10], allowing users to refine neural networks during and after training. It facilitates accuracy-aware optimizations to maintain performance while compressing models and offers post-training techniques like quantization to decrease model size and improve inference speed, particularly on edge devices with limited resources.
In this tutorial, attendees will learn as well how to optimize DL models using NNCF, as we showcase computer vision pipelines, such as Object Detection and Generative AI pipelines including Stable diffusion, LLMs, and multi-modal models. We will have demos for evidencing how OpenVINO runtime API is enabling real-time inference on laptops, edge devices, and resource-constrained hardware by more than 10x(_Performance varies by use, configuration and other factors. Learn more at [intel.com/performanceindex](intel.com/performanceindex)_)  in latency reduction for Stable Diffusion workloads. 


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

[11] K. Chen, C. Loy, H. Hu, H. Zhao, and H. Duan, "OpenMMLab: A Foundational Platform for Computer Vision Research and Production", [Online]. Available: https://openmmlab.com/community/cvpr2022-tutorial. OpenMMLab, 20 June 2022. [Accessed 27 November 2023].

[12] K. Chen, C. He, Y. Zeng, S. Zhang, and W. Zhang, "Boosting Computer Vision Research with OpenMMLab and OpenDataLab", [Online]. Available: https://openmmlab.com/community/cvpr2023-tutorial. OpenMMLab, 18 June 2023. [Accessed 27 November 2023].

[13] P. Ramos, Z. Wu, Y. Gorvachev, and R. Lo, "How to get quick and performant model for your edge application. From data to application", [Online]. Available: https://paularamo.github.io/cvpr-2022. Intel Corporation, 19 June 2022. [Accessed 27 November 2023].




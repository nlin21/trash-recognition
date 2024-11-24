# trash-recognition

Trash classification with YOLOv8

https://github.com/ultralytics/ultralytics  
https://github.com/pedropro/TACO

TODO:
- train model on TACO dataset
- integrate with external source sent over WiFi/BT

## Setup
1. Camera is required (Webcam, USB camera, etc.)
2. Install [Python](https://www.python.org/downloads/): check [here](https://github.com/ultralytics/ultralytics?tab=readme-ov-file#documentation) for the required version(s)
3. Install the ultralytics package with pip:

   ```
   pip install ultralytics
   ```
4. pip may install a CPU-only version of pytorch, which is lightweight but leads to poor inference times. I recommend running YOLOv8 on a dedicated GPU with CUDA support. To check the pytorch version:
      
   Run python, then enter
   ```
   import torch
   print(torch.__version__)
   ```
   ![image](https://github.com/user-attachments/assets/0356748b-30b3-4e75-9824-2899256c353b)  
   If the version has +cpu appended at the end, it is a CPU-only version. The code can be run at this point with `python main.py`, but if you wish to use CUDA continue to step 5.

5. To set up a GPU:
     
   First verify that your machine recognizes your GPU with `nvidia-smi`
   
   Uninstall the current pytorch version with
   ```
   pip uninstall torch torchvision torchaudio
   ```

   Then head to the pytorch [website](https://pytorch.org/get-started/locally/) to check which versions of CUDA Toolkit are supported in the 'Compute Platform' row.  
   1. Install your desired CUDA Toolkit version [here](https://developer.nvidia.com/cuda-toolkit-archive) (I went with 12.1)
   2. Install the corresponding pytorch version with the command in 'Run this command' (see below)
     
   ![image](https://github.com/user-attachments/assets/41efd81f-46b9-4654-850c-fde9423abb7a)  
     
   Once both CUDA Toolkit and pytorch has been installed, you can run the check from step 4 to verify:
   
   ![image](https://github.com/user-attachments/assets/24f552c2-98ae-4bfb-babb-4b323cefe070)

6. Setup is now complete, plug in your camera device and run `python main.py`

# Testing
![image](https://github.com/user-attachments/assets/798fd587-08e1-42f0-adbd-122eec2f9d1a)

![Screenshot 2024-11-24 at 1 39 24 PM](https://github.com/user-attachments/assets/5a2ee018-f0ac-47fd-9317-2777da5836f3)




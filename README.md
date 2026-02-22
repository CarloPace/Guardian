Step 1: Prepare the Hardware

    1.	Connect the Camera: Carefully lift the plastic clip on the Pi's camera port (CSI), insert the camera ribbon cable (with the metal contacts facing away from the Ethernet port), and push the clip back down.
    2.	Connect the Microphone: Plug USB microphone into any available USB port on the Raspberry Pi.
    3.	Power On: Turn on Raspberry Pi.

Step 2: Test the Hardware

    Open the terminal on Raspberry Pi and make sure the system recognizes sensors:
    •	Test the Camera: Type libcamera-hello and press Enter. A window should briefly pop up showing the live camera feed.
    •	Test the Mic: We will test this in Step 4 once the Python environment is ready.

Step 3: Set up the Project Folder

    need a place to store all  project files. Open terminal and do the following:
    1.	Create the folder (let's call it guardian):
    mkdir ~/Desktop/guardian
    cd ~/Desktop/guardian    
    2.	Add files: You can either download GitHub repository directly using git clone [_REPO_URL] . or manually transfer the prototype.py script and the .tflite model files into this guardian folder. Make sure everything is in this single folder.

Step 4: Create the Python Environment

    Modern Raspberry Pi software requires you to use a "virtual environment" to install Python packages safely.
    1.	Create the environment:
    python3 -m venv venv
    2.	Activate the environment:
    source venv/bin/activate
    3.	Install the required packages:
    pip install numpy==1.24.2 picamera2==0.3.31 requests==2.32.5 scipy==1.17.0 sounddevice==0.5.3 soundfile==0.13.1 tflite_runtime==2.14.0 opencv-python==4.6.0.66

Step 5: Confirm the Microphone Index

    1.	In the terminal (with venv still active), run:
    python3 -m sounddevice
    2.	This prints a list of audio devices. Find USB microphone and note the ID number next to it.
    3.	Open prototype.py in a text editor. If mic’s ID is not 1, change MIC_INDEX = 1 to whatever number you just found.
    
Step 6: Set up Telegram Alerts

    While you have prototype.py open:
    1.	Find the Telegram section at the top of the code.
    2.	Replace "NA" with actual Telegram Bot Token and personal Chat ID (which you can get by messaging @BotFather and @userinfobot on the Telegram app).
    3.	Save the file.
    
Step 7: Run the System!

    You are all set. Make sure you are in guardian folder and virtual environment is active.
    1.	Start the monitoring system:
    python3 prototype.py
    2.	The terminal will start printing updates. Stand in front of the camera or play a baby crying sound. If everything is working, the system will detect it and immediately ping phone with a Telegram message containing the photo or audio clip!


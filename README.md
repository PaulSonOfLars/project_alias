# Project Alias

<p float="left">
<img src="imgs/alias.jpg" width="49%"> <img src="imgs/short_alias_explained.gif" width="49%">
</p>

[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Project Alias is an open-source parasite to train custom wake-up names for smart home devices while disturbing their built-in microphone. Read more about the project [here](http://bjoernkarmann.dk/project_alias).

<p style='color:red'>NOTE: this project is experimental and still in development.</p>


## Build Guide


For the complete step-by-step guide and 3D files see the [Instructables](https://www.instructables.com/id/Project-Alias) by the original project creator, [Bjørn Karmann](https://github.com/bjoernkarmann).


## Raspberry Pi Setup 🔧

How to prepare and setup a Raspberry Pi for this project:

1. Download the latest version of [Raspbian](https://www.raspberrypi.org/downloads/raspbian/) and flash your micro SD card with [Etcher](https://etcher.io/), `dd`, or whichever preferred tool.

2. Copy the `ssh` and `wpa_supplicant.conf` files from the [setup folder](setup/) to the SD card's boot partition.

3. Edit the `wpa_supplicant.conf` in a text editor to match your wifi settings. Insert the card to the raspberry pi.

4. In a terminal, ssh into the pi: ```sudo ssh pi@<IP address>```<br>*Default password is 'raspberry'. To change password use the 'passwd' command*

5. Update the pi: ```sudo apt-get update && sudo apt-get upgrade```<br>

7. Reboot ```sudo reboot```


## Installing 


On the Raspberry Pi: clone and install the sound driver for the [ReSpeaker](http://wiki.seeedstudio.com/ReSpeaker_2_Mics_Pi_HAT/) hat:<br>
*This is only required when using the ReSpeaker hat, this code will also work with other sound drivers.*

```
cd && git clone https://github.com/respeaker/seeed-voicecard.git
cd seeed-voicecard && sudo ./install.sh
```

Install required **packages** (python modules are installed as packages, as pip3 may run out of memory on a pi 3A+):

```
sudo apt-get install python3-dev python3-pip git libatlas-base-dev
sudo apt-get install python3-numpy python3-spidev python-h5py python3-pyaudio libsdl-ttf2.0-0 python3-pygame
```

Install the required python modules:


```
sudo pip3 install tensorflow keras
sudo pip3 install flask flask_socketio python_speech_features
```

**Note**: If installing tensorflow via pip causes issues, install it by downloading the tensorflow`.whl` file and running

```
sudo pip3 install <path to tensorflow .whl>
```

Clone the **Alias** project: 

```
git clone https://github.com/PaulSonOfLars/project_alias.git
```

Setup a bootscript. Open this file:

```
sudo nano /etc/rc.local
```

and add at the end of the command just before **exit 0**, like:
  
```
cd project_alias && python3 app.py &
```
Now reboot the Pi to test it:

```
sudo reboot
```

## Training Alias 🍄


1. To train Alias use the browser on your phone and open ```raspberrypi.local:5050```

2. Hold down the record button while saying the new name about 4-6 times. A small bar should indicate the 2 seconds recording window. Each name should fit within this timeframe.

3. Under the menu, click **Train Alias** and wait a few seconds for the model to learn the name. This name does not necessarily need to be a word but can be a sound and any language. So be creative! You can always reset your name on the menu. *Tip: it helps to record the name from different locations in your home.*

4. Try it out! Say the name and ask your question once you see a blue light on the device or on your phone. 
Note: once trained there is no need to have the phone connected anymore. 

*If you find Alias is not responding correctly, try to train a few more examples. Or if you find Alias is triggering to often, you can go to the menu and turn background sound ON. This toggles the background mode and adds any new recordings to the background examples. Record and train just as before, but try to capture unique sounds in your environment or even words that sound similar to your chosen name.*

## Calibration 

- If you are using a **Amazon Alexa**, please change the `ASSISTANT` variable in `config.py` to `Alexa`.

- To set the volume of the speaker you can change the volume in `config.py` to another value.

## TODO:

- investigate if sudo is really necessary -> running as own user atm. (change in sound init function)
- remove requirement for rc.local script -> use systemd
- investigate potential weakness in "end of trigger" sentences; sometimes the word get constantly detected, even in silence.
- get rid of remaining globals:
    - has been trained -> classifier
    - reset -> state -> should be in webserver logic?
    - train -> state -> should be in webserver logic?
    - predict -> state -> should be in webserver logic?
    - BG_examples -> classifier
    - TR_examples -> classifier
    - triggered -> main loop
    - example ready -> what is this
    

## Get Involved!
We are both Interaction Designers, Makers and strong believers in privacy but no experts when it comes to Speech Recognition software. If you are interested in getting involved in version 2.0 please let us know!

## Contributors
Made by [Bjørn Karmann](http://bjoernkarmann.dk) and [Tore Knudsen](http://www.toreknudsen.dk/).
Extended by [Paul Larsen](https://pnvlarsen.me).


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


# Gstreamer-Realtime-Acquition

## Installation

### Install GI (PyGObject) Library (Better to install in Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav
```

### Create Virtual Environment

## Option 1: Create Virtual Environment (venv)

```bash
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install -r requirements.txt
```

## Option 2: Create Conda Environment

```bash
conda create -n gstreamer python=3.10
conda activate gstreamer
conda install -c conda-forge pygobject gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly
pip install -r requirements.txt
#CHANGE USER TO CURRENT USERNAME
sudo cp -r /usr/lib/x86_64-linux-gnu/gstreamer-1.0/* /home/USER/miniconda3/envs/gstreamer/lib/gstreamer-1.0/

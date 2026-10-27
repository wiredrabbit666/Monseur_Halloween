# Monseur_Halloween

This is a fun little application of image recognition and face tracking.  It shows a pair of glowing cat eyes on your screen that follow whoever is standing in front of the webcam.  Just the creepy thing for a screen in a window or at a halloween party.

The pupils narrow to slits as someone gets closer or walks toward the camera, the eyes blink now and then, and they wander around when nobody is there. Everything runs on your own computer; nothing is sent to the internet.

It works by running a small, ready-made AI model (`yolo26n-pose.pt`, included in this repo) that finds people and their body positions in each webcam frame. You don't need to know anything about AI or train anything; the model is used as-is.

Eventually I'll add more functionality and make it friendly for non-python peoples.

## What you need

- A Windows PC with a webcam (macOS and Linux should also work but I've not tested them; see the notes in each step)
- A processor with 4 or more cores from the last 5–6 years. Older ones work too; the eyes just
  react a little later.
- 8 GB of memory recommended (the app itself uses about 1–2 GB)
- About 1 GB of free disk space, or 3.5 GB for the NVIDIA install (the AI libraries are large)
- Optional: an NVIDIA graphics card takes load off the processor, but it is not required; the eyes run smoothly without one.  I have problems with 4K multi-screen and MSI KVM using GPU.

## Step 1: Install Python

1. Download Python 3.12 or newer from <https://www.python.org/downloads/>.
2. Run the installer. On the first screen, **tick "Add python.exe to PATH"** before clicking Install.
3. Open a new terminal (press the Windows key, type `PowerShell`, press Enter) and check it worked:

   ```
   python --version
   ```

   It should print something like `Python 3.12.x`.

## Step 2: Get the code

Either download it: on the GitHub page click the green **Code** button, then **Download ZIP**, and unzip it somewhere such as your Documents folder.

Or, if you have Git installed:

```
git clone https://github.com/wiredrabbit666/Monseur_Halloween.git
```

Then, in the terminal, go into the folder (adjust the path to where you put it):

```
cd Documents\Monseur_Halloween
```

## Step 3: Install the libraries

This creates a private "virtual environment" (a folder called `.venv`) so the libraries don't
interfere with anything else on your computer:

```
python -m venv .venv
.venv\Scripts\activate
```

Your prompt should now start with `(.venv)`. On macOS/Linux use `source .venv/bin/activate` instead.

If Windows says running scripts is disabled, run this once and try the activate line again:

```
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Now install **one** of the two sets. This downloads a few gigabytes and can take several minutes.

- **No NVIDIA graphics card, or not sure:** works on any computer

  ```
  pip install -r requirements.txt
  ```

- **NVIDIA graphics card:** faster. Update your NVIDIA driver first (GeForce Experience or
  <https://www.nvidia.com/drivers>), then:

  ```
  pip install -r requirements-gpu.txt
  ```

  To check the card is detected (should print `True`):

  ```
  python -c "import torch; print(torch.cuda.is_available())"
  ```

## Step 4: Run it

```
python cat_eyes.py
```

The eyes open fullscreen. Step in front of the webcam and they should follow you.

If you installed the NVIDIA set, run on the graphics card with:

```
python cat_eyes.py --device gpu
```

If no usable NVIDIA card is found it just prints a note and runs on the processor instead.

**Next time**, you only need to open a terminal in the folder, activate, and run:

```
.venv\Scripts\activate
python cat_eyes.py
```

### Keys

| Key   | Action                                             |
|-------|----------------------------------------------------|
| `q`   | Quit                                               |
| `Esc` | Leave fullscreen (press again in a window to quit) |
| `f`   | Toggle fullscreen                                  |
| `v`   | Show the webcam view with detected people          |
| `m`   | Mirror the gaze (if the eyes look the wrong way)   |

### Options

| Option            | What it does                                           |
|-------------------|--------------------------------------------------------|
| `--device gpu`    | Use the NVIDIA graphics card (default is `cpu`)        |
| `--windowed`      | Start in a window instead of fullscreen                |
| `--camera 1`      | Use a different webcam (0 is the first one)            |

To show the eyes on a second screen: start with `--windowed`, drag the window onto that screen,
then press `f`.

## Troubleshooting

- **`python` is not recognized:** Python wasn't added to PATH. Re-run the installer, choose
  *Modify*, and tick *Add Python to environment variables*, then open a new terminal.
- **Black window or no camera:** another app (Teams, Zoom, the Camera app) may be using the webcam;
  close it. With several webcams, try `--camera 1`.
- **The eyes don't follow anyone:** press `v` to see what the camera sees. Make sure the room is lit
  and your upper body is in view.
- **Second screen goes black when using `--device gpu`:** some monitor/laptop setups don't like it.
  Run without `--device gpu`; the processor is fast enough.

## For developers

`run_demo.py` shows the raw tracking: every detected person with their importance score, plus
sliders to tune it. The tracking code lives in the `focus/` folder.

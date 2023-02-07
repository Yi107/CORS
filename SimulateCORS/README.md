# Description of SimulateCORS

This folder contains the code of simulating cooperative route planning based on the CORS frame work in Section 4 in paper Cross Online Ride-sharing for Shared Mobilities in Spatial Crowdsourcing. The experimental environment is based on python3.9 and Wolfram application.

## Pre-setting

You need to install the Wolfram Client Library for Python first. https://reference.wolfram.com/language/workflow/InstallTheWolframClientLibraryForPython.html.en?source=footer

There are two ways to setup a session of wolfram: based on local kernel and based on cloud engine.

### Local kernel setting

You can download Wolfram Engine on https://www.wolfram.com/engine/index.php.en?source=nav&source=footer. You can find the kernel path on https://reference.wolfram.com/language/WolframClientForPython/docpages/basic_usages.html#expression-representation. 



### Cloud Engine setting

You can start an authenticated session with the Wolfram Cloud:https://reference.wolfram.com/language/workflow/ConnectPythonToTheWolframCloud.html.en?source=footer



## Description of **"config.py"**

 **config.py** contains the basic settings of the algorithms in this folder. All the urls and parameters can be modified according to personal preference. 

## Description of "Simulator.py"

**Simulator.py** is the main program to simulate CORS. You can run this file to perform two algorithms we proposed in the paper: R-CORS and D-CORS by defining the parameter "mode" in this file.

The input of this program is the order file, the gps file, and the driver files, which can be modified in "config.py". The outputs of this program are the working status diagram of the ride-sharing platforms, and the final running results. The working status diagram includes workers and requests distribution, as well as the distribution of rejected requests.



## Description of other python files

Other python files are the strutures of simulator.py. The description of these files are shown in the python files.

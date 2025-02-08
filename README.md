# General Repo for code handling of algorithmic trader

repo consists of:
* data downloader to create datasets
* Kucoin Api library focusing on marging and high-frequency margin trading
* Algo Trader script that executes trade strategies and translates them to the kucoin API

# Work in progress
* trade strategies
* correct balance tracking
* Algo Trader finalisation 

# install venv

## for packages
1. Double click on the install_venv.bat and the console should automatically. install all compatible libraries and packages.

### Errors with installation
* If you spot a library that is currently missing, please add to the requirements file, upload to github, and re-install the venv again. 
* It might be that python was not added to your system environments. Then just add the python.exe to your system variables as admin and it should work. 

## for tensorflow-gpu:

1. install Visual Studio 2022 from Microsoft to fully comply with install requirements for tensorflow

* https://visualstudio.microsoft.com/de/

2. install the below packages for NVIDIA cuda from here:

select correct compatible versions from here https://www.tensorflow.org/install/source#gpu

you need version 11.2
* https://developer.nvidia.com/cuda-downloads
* https://developer.nvidia.com/cuda-11.2.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal

You need version 8.1
* https://developer.nvidia.com/cudnn-downloads
+ https://developer.nvidia.com/rdp/cudnn-archive

2. then install tensorflow-gpu using the terminal

## installation through WSL (Linux) 

Running tensorflow > 2.1v requires cuda and cudann to be installed through WSL. This also means that a seperate venv needs to be installed and activated through WSL. 

When WSL is installed do the below:

1. run the bat setup_gpu_wsl.bat
2. Open WSL terminal and activate the venv: source ~/path/to/repo/scripts/activate_gpu_venv.sh


## Google Drive API Setup

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a project and enable the **Google Drive API**.
3. Create **OAuth 2.0 Client ID** credentials and download the `credentials.json` file.
4. Place the `credentials.json` file in the root directory of the project.
5. Ensure `credentials.json` is listed in `.gitignore` to avoid uploading it to version control.

## Google Drive API Setup
The repository also supports integration with Google Drive for storing and retrieving files such as logs, CSVs, and plots.

OAuth2 and Service Account Authentication
You can authenticate using either OAuth2 (for user login) or Service Account (for server-side automation). Here’s how to set up both:

### Using OAuth2 for User Authentication
1. Go to the Google Cloud Console.
2. Create a new project and enable the Google Drive API.
3. Create OAuth 2.0 Client ID credentials:
3. * Choose Desktop App as the application type.
3. * Download the credentials.json file.
4. Move credentials.json to the root directory of your project.
5. Ensure that credentials.json is added to .gitignore to prevent it from being uploaded to version control.

#### First-time OAuth2 Authentication:
When you run the script for the first time, it will open a browser window and ask you to log in to your Google account. The OAuth token will be saved as token.json for future runs, preventing the need for repeated logins.

google_drive_util = GoogleDriveUtility(credentials_type="oauth2", credentials_path='repoFolder/credentials.json', token_path='repoFolder/token.json')

### Using Service Accounts for Automation
If you want the script to run autonomously, use Google Service Account Authentication. Here’s how:

1. Step 1: Create a Service Account
1. * Go to the Google Cloud Console.
1. * Navigate to IAM & Admin > Service Accounts.
1. * Create a new service account and assign the necessary roles (e.g., Editor or Owner).
1. * Download the Service Account Key as a JSON file and move it to your project folder (e.g., repoFolder/service_account_key.json).

2. Step 2: Share Your Google Drive Folder
2. * Go to Google Drive and create the folder where your service account will store files (e.g., Algotrader_Files).
2. * Share the folder with your service account's email address (found in the service_account_key.json file). Give it Editor access.

3. Step 3: Modify the Script for Service Account Authentication
Update your script to use the service account for Google Drive API interactions:

google_drive_util = GoogleDriveUtility(credentials_type="service_account", credentials_path='repoFolder/service_account_key.json')

#debugging in WSL and VS code
1. go to the .vscode folder and copy the details from launch_wsl to launch.json if you want to debug in wsl, otherwise copy from launch_windows to launch.json

# Authors
* TheMoMatthas
* Janson Ericson

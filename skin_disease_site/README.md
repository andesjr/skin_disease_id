# minimal_image_project

This package contains the minimal structure to push a typical image classification project in production

## Structure :

We propose a two-services structure:
- the backend will run in a container on google cloud run
- the frontent will run on heroku

### Backend part:
- the code to train the model is in `minimal_image_project_back/minimal_image_project_back`
  - the basic provided model is a tensorflow ConvNet, with binary classification as output
- the code for the api is in `minimal_image_project_back/api`
  - the api is managed by fastapi
  - the api is served by a uvicorn server
### Frontend part:
- streamlit page defined in `minimal_image_project_front`
- the deploy method will be through github to heroku, but without using CD actions. We will prefer the automatic 'pull' deploy mode proposed by heroku.


## Steps to reproduce the deployement :
Follow thse steps to reproduce the toy example provided, and make the necessary adaptations to your project

*Nota* : most of the commands are available in the makefiles of each part


|step|instructions   | notes      | adaptations |
|---|---|---|---|
|Clone this repo | `git clone https://github.com/bdepebhe/minimal_image_project.git`<br /> `cd  minimal_image_project`| ||
|Remove git from the folder | `rm -rf .git` | Since you will need to push the frontend sub-directory with git, the parent folder containing back and front should not be a git directory|
|Install back end dependencies locally | `(cd minimal_image_project_back && pip install -r requirements_docker.txt)`| tensorflow is not part of `requirements_docker.txt`, so it is safe to trigger this command even on a mac M1 where the installation of tensorflow is custom (for example `pip install tensorflow-macos==2.5.0 tensorflow-metal==0.1.2`)| Dont forget to add other specific required package to `requirements_docker.txt`. Only those used in the container at prediction time.|
|Train the model   | `(cd minimal_image_project_back && make train_model)` | | Your model will probably trained by a dedicated piece of code on another machine, such as on google colab or other. In this case, just copy the fitted model (see https://www.tensorflow.org/tutorials/keras/save_and_load)<br /> Alternatively you can use the .h5 format |
|Run the API locally | `(cd minimal_image_project_back && make test_api_loccally)` | | Edit `api/fast.py` : change the location of the model (might be a h5 file), or the prediction output format.|
|Test root endpoint of API locally | http://localhost:8000/ and check that the json received contains the greeting message. Keep the API running by using a new terminal tab|
|Install front end dependencies locally | `(cd minimal_image_project_front && pip install -r requirements.txt)`| | Dont forget to add other specific required package to `requirements.txt`. Anything imported in your `streamlit_app.py` file.|
|Run the Streamlit page locally | `(cd minimal_image_project_front && make run_streamlit_locally_with_api_on_port_8000)`| | Create your custom frontend. For example, after receiving the result from API, you might want to display it your own way|
|Test the complete stack front+back locally| http://localhost:8501 and check that the streamlit page is displayed. Then upload any jpg or png image, and check that the prediction is displayed, for example `1.0`| | Make your own manual test and iterate other both the API code and the page design. They are both in autoreload mode, so you can make fast loops ! |
|Enter your GCP project name | edit `PROJECT_ID` variable in `minimal_image_project_back/Makefile`| | If necessary : modify also `IMAGE_NAME` and `GCR_REGION`|
|Build your docker image | on wsl or linux or mac intel : `(cd minimal_image_project_back && make build_image_on_linux_amd_for_any_platform_except_m1)`<br /><br /> on mac M1 : `(cd minimal_image_project_back && make build_image_on_mac_m1_for_cloud_run)` | On mac M1, we use a special builder (buildx) and must specify the amd64 architecture (google servers are amd64 machines), because the default `docker build` command makes images for the local architecture (arm for the mac M1 machines)| Edit the dockerfile if other files are needed inside the image. Note that the tensorflow installation has been put on purpose on the top layer, because this is a heavy step that can be cached in case of successive multiple builds |
|Test your docker image locally | on wsl or linux or mac intel : (`cd minimal_image_project_back && make test_container_locally_on_linux_amd)` then http://localhost:8000/ and check that the json received contains the greeting message <br /><br /> on mac M1 : not possible | A workaround for M1 would be to use a arm image built just for the purpose of this test (using the instruction `build_image_on_mac_m1_for_mac_m1_warning_not_possible_if_tensorflow` in the Makefile). But for the moment (jan.2022), it seems that Tensorflow has not released a version capable of the arm architecture and the linux buster OS., so the build raises an error when trying to pip install tensorflow|
|Push docker image to GCP | `(cd minimal_image_project_back && make push_image_to_gcr)`| Google Cloud Registry might be shortly depreciated. See Artifact Registry |
|Deploy a container on cloud run | `(cd minimal_image_project_back && make deploy_container_to_cloud_run)` and follow instruction. **ACCEPT `Allow unauthenticated invocations`** <br /> Note the Service URL|
|Test local front with cloud backend | Copy the Service URL as variable `SERVICE_URL` in `minimal_image_project_front/Makefile`<br /> Then `(cd minimal_image_project_front && make run_streamlit_locally_with_api_on_cloud)` <br /> http://localhost:8501 and check the product with an image| | 
|Push front to github| `(cd minimal_image_project_front && git init && gh repo create)` and follow instructions <br /> Then `(cd minimal_image_project_front && git add * && git commit -m 'initial commit' && git push origin master)`|
|Create an app on heroku and configure the app for autodeploy| https://heroku.com/, login and create an app<br /> In your app, choose `Deploy > Deployment method > Github`. Follow instruction to allow heroku to access your github <br /> Then In `Connect to GitHub` select your GihHub repo, <br /> Select `Automatic deploys > Enable Automatic Deploys`. <br /> Make your first deploy with `Manual deploy > Deploy Branch`| We use manual deploy for the first deploy, but we could alternatively make a useless push to master on github, with a useless change like a trailing line in any file. Since the autodeploy is activated, this would automaticly deploy on heroku.|


# Fatique-Detection-From-Physiological-Signals

[![Python application test with Github Actions](https://github.com/el-dAna/Fatique-Detection-From-Physiological-Signals/actions/workflows/main.yml/badge.svg)](https://github.com/el-dAna/Fatique-Detection-From-Physiological-Signals/actions/workflows/main.yml)

## Data
The  first things is to host the data reliably, transfrom it and make it ready for inferencing using the developed models. This is not done on github but on aws.
1. First an s3 bucket is created. US West (N. California) us-west-1 servers used.
ACLs was not enabled .(Objects in this bucket can be owned by other AWS accounts. Access to this bucket and its objects can be specified using ACLs.)
2. Data uploaded
3. Custom preprossing done using lambda and step funcions.

`docker build -t tag_name .`
`docker image ls`
`docker run -p 127.0.0.1:8080:8080 image_id`
`docker rmi image_id --force`
`docker rmi $(docker images -q) --force`
`docker stop container`


started docker image build but encountered errors. 


Challenges.
Running a standalone streamlit app was smooth. Packaging and running docker image had some session states apparently not initialised! uff! I changed the running port to default streamlit 8501 and it worked. 

authenticating codespaces with ecs in order to build and push image. codepaces is not a non TTY device. So i used local env to communicate with my ecs repositories. Cloud9 was not used because space on the free tier devices was not enough for pakages in requirements.txt. I got "WARNING! Using --password via the CLI is insecure. Use --password-stdin." Will use IAM role/policies in future. 
The solution was to create IAM role and attach needed policies(full access to ecs registry), then save the ID and secrete access keys of the role by running aws config on cli. Then after commands from the private ecs repository were run. Boom! To communicate with s3 bucket I added corresponding policy to the same IAM role created for ecs access. I just added full access to s3 bucket policy. I will change this later to readonly when published.

Deploy container to AWS Apprunner. Check that the port specified in the docker file is same in app runner config.

## Running inference
The idea is for users to train a new model
[change model parameters like input shape, dropout rates, loss, optimiser, epochs, train steps....] and data processing options like sampling window, degree of overlap beteen samples. Users can also determine percent of train data to use. All this via an api on the fly and see results on clearml
Save it to clearml / s3 bucket
And reload the model
Run inference. 

I think saving on clearml is more practical since models on s3 would have to be downloaded and saved explicitly in a file system. Clearml handles all this by calling a method. Will fully swith to clearml

Tagged model selection
Structure of trained models differ in their input shape based on the sample window selected during training. The input shape of architecture changes to accomodate the sample window. So it is important for users to be able to choose such models and process the subject data accordingly and then use uplaoded model for inference. All models are loaded if no compatible model is found and inference made using default specifications (sampling_window=100, overlap=0.5) 

## Next session
present file descriptions in tabular form, heading(file names), rows(recorded info, frequency, #samples, ....)
tag uplaoded models based on selected parameters
smooth linkage b/n pages. Currently some part of data preprocessing has to work before inference page works smoothly

---------------------------------------------------------------
## TO DOs before full deployment
put physilogicalsignalsbuket in a global variable, preferably in a dataclass in app.py



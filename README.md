# Fatique-Detection-From-Physiological-Signals

[![Python application test with Github Actions](https://github.com/el-dAna/Fatique-Detection-From-Physiological-Signals/actions/workflows/main.yml/badge.svg)](https://github.com/el-dAna/Fatique-Detection-From-Physiological-Signals/actions/workflows/main.yml)

## Data
The  first things is to host the data reliably, transfrom it and make it ready for inferencing using the developed models. This is not done on github but on aws.
1. First an s3 bucket is created. US West (N. California) us-west-1 servers used.
ACLs was not enabled .(Objects in this bucket can be owned by other AWS accounts. Access to this bucket and its objects can be specified using ACLs.)
2. Data uploaded
3. Custom preprossing done using lambda and step funcions.

`docker build -t image_tag .`
`docker image ls`
`docker run -p 127.0.0.1:8080:8080 image_id`
`docker rmi image_id --force`
`docker stop container`

started docker image build but encountered errors. 


Challenges.
Running a standalone streamlit app was smooth. Packaging and running docker image had some session states apparently not initialised! uff! I changed the running port to default streamlit 8501 and it worked. 
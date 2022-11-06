pip list --format=freeze > requirements.txt
docker build -t recomm_sys_beta_img .
docker run -d --name recomm_sys_beta_container -p 8080:8080 recomm_sys_beta_img
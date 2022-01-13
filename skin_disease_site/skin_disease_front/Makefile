# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

SERVICE_URL='https://skincancer-r5elmwbdhq-ew.a.run.app'

run_streamlit_locally_with_api_on_port_8000:
	streamlit run streamlit_app.py http://localhost:8000

run_streamlit_locally_with_api_on_cloud:
	streamlit run streamlit_app.py ${SERVICE_URL}

#replace by your own app that you can create on the web interface
HEROKU_APP_NAME=lewagonb779skincancer

create_remote_to_heroku:
	heroku git:remote -a ${HEROKU_APP_NAME}

push_to_heroku:
	git push heroku master

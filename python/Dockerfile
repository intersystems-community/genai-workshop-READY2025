FROM python:3.13.3-bullseye

WORKDIR /home/python/work

COPY . .

RUN pip install -r requirements.txt

CMD ["tail", "-f", "/dev/null"] 

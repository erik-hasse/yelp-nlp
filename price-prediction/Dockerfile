from continuumio/miniconda3
WORKDIR /app

COPY conda.yaml /app

RUN conda env create -f conda.yaml

COPY src/review_app.py resultsconfig.json resultstf_model.h5 entry.sh /app/

ENTRYPOINT ["./entry.sh"]

FROM rocker/r-ver:4.4.1

WORKDIR /app

COPY /src/r/install_packages.R /app/src/r/install_packages.R
COPY /src/r/titanic_pipeline.R /app/src/r/titanic_pipeline.R

RUN Rscript /app/src/r/install_packages.R

CMD ["Rscript", "/app/src/r/titanic_pipeline.R"]

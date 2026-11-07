# Assignment 2 — Run and Submission Guide

## 1. What to submit
The assignment brief requires ONE PDF, in this order:
1. Public GitHub repository URL.
2. Live Streamlit Community Cloud app URL.
3. One screenshot showing execution on BITS Virtual Lab.
4. The complete README content, including dataset description, metrics table,
   model observations and overall winner.

The GitHub repository itself must include:
- app.py
- train_models.py
- requirements.txt
- README.md
- test_data.csv
- model/ containing all five saved `.joblib` files
- model_metrics.csv (useful supporting result file)
- dataset_info.json

## 2. Run locally or in BITS Virtual Lab
Open a terminal in the project directory and run:

    pip install -r requirements.txt
    python train_models.py

This generates test_data.csv, model_metrics.csv, dataset_info.json and the
five saved models in model/. Open model_metrics.csv and copy the real values
into README.md. Review the observations against the actual values.

Then launch:

    streamlit run app.py

Upload test_data.csv and test each of the five choices. Capture a screenshot
of the assignment execution inside BITS Virtual Lab for the required mark.

## 3. GitHub
Create a new public repository, for example `aiml-ml-assignment-2`.
Upload the project files and folders, commit them, and verify that the files
are visible on GitHub. The assignment asks for a clear README and commit
history. Do not submit a repository that you have not personally checked.

## 4. Streamlit Community Cloud
Sign in at https://streamlit.io/cloud using GitHub. Choose New app, select your
repository, branch `main`, and `app.py`, then deploy. The model files and
test_data.csv must be committed to the repository. Open the deployed URL in
a private/incognito browser window and test the upload and all five models.

## 5. Final PDF
Create one PDF containing the two working links, the BITS Lab screenshot,
and the completed README content. Replace every placeholder. Do not submit
until the live app works and the screenshot is included.

## Important
This project is a working starting point for learning and adaptation. The
assignment brief explicitly says AI tools are allowed for learning support,
not direct copy-paste submissions. Run the code yourself, inspect the outputs,
write observations based on your actual metrics, customize the app, and be
prepared to explain your implementation.

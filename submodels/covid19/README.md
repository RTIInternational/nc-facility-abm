# COVID submodel

## Setup / Tests / Profiling

### Environment Setup
This submodel has dependencies in addition to those of the main `nc-hospital-abm` model. 

After setting up a virtual environment per the instructions in the root README, activate the environment and run the following command to install these additional dependencies.

```
pip install -r submodels/covid19/requirements.txt
```

### Tests / Profiling

For standard command line output:

```bash
pytest submodels/covid19/model/ --cov=submodels/covid19/model --verbose
```

**Profiling:**

```bash
sudo pyinstrument --html -o profiletest_covid.html submodels/covid19/src/profile_run.py
```

## Collecting Input Data/Parameters (Cases/Vaccinations/Staffing)

### Parameters

**Hospitalization Parameters**

To estimate hospitalizaiton probabilities, run the following script. These values should be added to the `COVIDParameters`.

```
python submodels/covid19/src/cases_to_probabilities.py 
```

**Effective Reproductive Number**

To calibrate `re` for a specific time period, you can work through `submodels/covid19/seir/estimate_re.py`. This will try different `re` values and compare the SEIR output to known case counts.


### Input Data

**Latest COVID-19 Cases:**

You can register for a key [here.](https://apidocs.covidactnow.org/#register)

```bash
python submodels/covid19/src/download_cases.py {your_api_key}
```

**NC COVID-19 Vaccinations**

- Go to the NC DHHS Data Website: [link](https://covid19.ncdhhs.gov/dashboard/data-behind-dashboards)
- Look for "Data Behind the Dashboard".
- Find the "People Vaccinated Demographics" tab.
- Select "Age Group" and "People fully vaccinated"
- Download this file and place it here: `submodels/covid19/data/vaccinations/vaccinations_by_age.csv.csv`

Run:
```
python submodels/covid19/src/create_vaccination_rates.py
```

**Latest COVID-19 Vaccinations (Not currently used)**

No API key needed - Ignore the warning related to an API key when running this script.

```bash
python submodels/covid19/src/download_vaccinations.py 
```

**Payroll Based Journal nursing home staffing data**

In addition to the CMS data used in the base model, this submodel uses another CMS dataset called Payroll Based Journal (PBJ) to calculate staff hours per nursing home per week.

The PBJ data is available from the CMS website and is separated into two files:
- [Daily Nurse Staffing](https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing)
- [Daily Non-Nurse Staffing](https://data.cms.gov/quality-of-care/payroll-based-journal-daily-non-nurse-staffing)

PBJ data is released quarterly. The first version obtained for this project covers Q2 2021 and was published in October 2021, suggesting an approximate lead time of 2 quarters.

The base data files are not stored in this repo, since they are large.

To update the PBJ data:
- download the base files from the links above. Save them in `submodels/covid19/data/pbj`
- Update the file paths in `submodels/covid19/config/filepaths.yaml` if necessary
- run `python submodels/covid19/src/process_pbj_data.py` to process the base files and save an updated version of `submodels/covid19/data/pbj/PBJ.csv`.

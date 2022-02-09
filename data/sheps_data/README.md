# Convert PDFs to CSVs

The necessary data for the CDC project is exported in PDFs that contain both tables and text. Since there are hundreds of these tables, a script is necessary to extract the data. Follow these steps to replicate the process for 2018. The column names will need to be updated with future years.

## Get PDFs

Download the two relevant pdfs for CDC HAI. [1](https://www.shepscenter.unc.edu/wp-content/uploads/2020/05/ptchar_all_and_by_hosp_2018_and.pdf) [2](https://www.shepscenter.unc.edu/wp-content/uploads/2020/05/ptorg_pt_res_by_hosp_2018.pdf) [3](https://www.shepscenter.unc.edu/wp-content/uploads/2020/05/discharge_all_and_by_hosp_2018_and.pdf)

Move both files to here (this directory).

## File 1 Patient Characteristics

### Initial Processing of Patient Characteristics by Hospital (ptchar)

Run this script to extract the data. This script only works for hospitals that reported data for the last five years. A list of hospitals that fail this criteria will be created.

```
python data/sheps_data/src/read_ptchar_pdf.py
```

### Manually Evaluate Hospitals and Add As Needed to Processing File


For all hospitals in `<year>_manual_evaluation_ptchar.txt`, manually evaluate whether they reported data in the most recent year. If they did, note the page numbers that were not processed and add these page numbers to the list of pages in: `data/sheps_data/src/add_outlier_hospitals.py`.

### Final Processing of ptchar

After manually adding the page numbers, run the following:

```
python data/sheps_data/src/add_outlier_hospitals.py
```

### Create CSV Subset

To capture the subset of variables used, after step 4, run the following:

```
python data/sheps_data/src/make_ptchar_subset.py
```

This will `create subset_ptchar_for_analysis.csv`. It will also create a total column and check for differences between expected and calculated output. If a warning is printed, one of the tables was not extracted correctly.

### File 2: Patient County of Residence

Process the ptorg file with the following:

```
python data/sheps_data/src/read_ptorg_pdf.py
```

Note that the Actual total and Calculated total do not always match as the PDF values do always add up to 100%. The difference is made into a new row called Unreported.


### File 3: Total Admissions and LOS by Hospital

```
python data/sheps_data/src/read_discharge_pdf.py
```




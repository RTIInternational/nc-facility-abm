import camelot
import numpy as np
import pandas as pd
import pdfquery
from data.sheps_data.src.helpers import ptchar_var_map
from data.sheps_data.src.read_ptchar_pdf import file, main_dir, n_year, pdf_cols, rows_to_drop, year
from tqdm import trange

if __name__ == "__main__":
    # default pages specify rows that are used every time
    default_pages = [1, 2, 3]
    # add additional pages based on manual_evaluation_ptchar.txt
    page_list = default_pages + [40, 41, 42, 94, 95, 96, 154, 155, 156, 214, 215, 216, 289, 290, 291, 337, 338, 339]

    # query pdf for hospital titles
    pdf = pdfquery.PDFQuery(file)

    final = pd.read_csv(main_dir.joinpath(f"{year}_master_ptchar_initial.csv")).set_index(["variable"], drop=True)
    master_df = pd.DataFrame()

    # for each table (assuming 1 per page) determine the hospital and extract
    # necessary data
    for i in trange(len(page_list)):
        # read pdf for tables
        table = camelot.read_pdf(str(file), flavor="stream", pages=str(page_list[i]))

        # extract the hospital name for that page
        pdf.load(page_list[i] - 1)
        label = pdf.pq('LTTextLineHorizontal:contains("Short Term Acute Care Hospital Discharge Data")')
        left_corner = float(label.attr("x0"))
        bottom_corner = float(label.attr("y0"))
        hospital = pdf.pq(
            'LTTextLineHorizontal:in_bbox("%s, %s, %s, %s")'
            % (left_corner, bottom_corner - 30, left_corner + 500, bottom_corner)
        ).text()

        hospital_string = hospital.replace(" ", "_").lower()

        # extract the dataframe for that page
        df = table[0].df
        df.rename(columns={df.columns[0]: "variable"}, inplace=True)

        # this pdf has 3 types of tables determine the type of page by locating appropriate headers
        if df["variable"].str.contains("Patient Distance to Hosp").any():
            page_type = 1
            keep = [i not in ["Patient Characteristics", "", "Patient Residence State", "Unknown"] for i in df.variable]
        elif df["variable"].str.contains("Payer").any():
            page_type = 2
            keep = [i not in ["", "Point of Origin", "Patient Characteristics", "Unknown"] for i in df.variable]
        elif df["variable"].str.contains("Home, self, or outpatient").any():
            page_type = 3
            keep = [i not in ["", "Patient Characteristics", "All", "Patient Disposition"] for i in df.variable]
        else:
            raise print("Unknown page type")

        df = df[keep]
        df = df[~df["variable"].isin(rows_to_drop)]

        # replace blank spaces with nans
        df = df.replace("", np.nan)
        # replace new line symbols with spaces
        df.replace({"\n": " "}, regex=True, inplace=True)

        if df.shape[1] == 2:
            df[2] = df[1].apply(lambda x: x.split()[1])
            df[1] = df[1].apply(lambda x: x.split()[0])

        cols = pdf_cols
        if len(list(df)) == 9:
            cols = pdf_cols[0:9]
        elif len(list(df)) == 7:
            cols = pdf_cols[0:7]
        elif len(list(df)) == 5:
            cols = pdf_cols[0:5]
        elif len(list(df)) == 3:
            cols = pdf_cols[0:3]

        # rows with na indicate that the columns did not separate correctly
        # when the table was read in from the pdf - identify these rows to separate the strings
        if len(list(df)) in [11, 9, 7, 5, 3]:
            rows_with_null = df[pd.isnull(df).any(axis=1)].set_index("variable")
            df.columns = cols
        # sometimes the pdf does not correctly read in the columns if the number of columns is not expected (ie not 11)
        # all rows must be split into 11 columns
        else:
            rows_with_null = df.set_index("variable")
            df = pd.DataFrame(columns=cols)

        # used later to drop rows
        rows_to_split = rows_with_null.index.values.tolist()

        df_temp = pd.DataFrame()

        # separate strings into columns
        # this dataframe contains many nas and the size is not predictable
        for col in rows_with_null.columns:
            if not rows_with_null[col].isnull().all():
                new_df = rows_with_null[col].str.split(" ", expand=True)
                df_temp = pd.concat([df_temp, new_df], axis=1, sort=False)

        df_append = pd.DataFrame()

        # remove nas from temporary dataframe
        for idx, row in df_temp.iterrows():
            new_row = row.dropna().reset_index(drop=True)
            new_row.columns = cols
            df_append = df_append.append(new_row, sort=False)

        df_append = df_append.reset_index()
        df_append.rename(columns={df_append.columns[0]: "variable"}, inplace=True)

        # if there are rows to append, add them to the dataframe
        if len(df_append) > 0 and df_append.shape[1] in [11, 9, 7, 5, 3]:
            df = df[~df["variable"].isin(rows_to_split)]
            df_append.columns = cols[0 : df_append.shape[1]]
            df = df.append(df_append, sort=False)

        df = df.reset_index(drop=True)

        var_map = ptchar_var_map

        if page_type == 1:
            var_map.update(
                {
                    "Other": "Patient Residence State Other",
                    "Missing": "Patient Dist to Hospital Missing",
                    "Unavailable": "Race Unavailable",
                    "Unknown": "Ethnicity Unknown",
                }
            )
        if page_type == 2:
            var_map.update({"Other": "Payer Other", "Unknown": "Payer Unknown"})

        df.variable = df.variable.map(var_map).fillna(df["variable"])

        df.columns = cols

        df["hospital"] = hospital

        mini_df = df[["variable", n_year, "hospital"]]

        master_df = pd.concat([master_df, mini_df], axis=0, sort=False)

    final_add = master_df.pivot_table(
        index="variable", columns="hospital", values=n_year, aggfunc=lambda x: " ".join(x)
    )
    final_add = final_add.replace("", np.nan)
    final_add = final_add.drop(columns=["Summary Data for All Hospitals"])

    final = pd.concat([final, final_add], axis=1)

    final = final.loc[:, ~final.columns.duplicated(keep="last")]

    final.to_csv(main_dir.joinpath(f"{year}_master_ptchar_final.csv"))

import camelot
import numpy as np
import pandas as pd
import pdfquery
from data.sheps_data.src.read_ptchar_pdf import main_dir, year
from tqdm import trange

keep_columns = ["Normal Newborns", "Excluding Normal Newborns"]


if __name__ == "__main__":
    # specify file and page range here
    file = main_dir.joinpath("discharge_all_and_by_hosp_2018_and.pdf")
    page_range = range(1, 120)
    page_list = list(page_range)

    # query pdf for hospital titles
    pdf = pdfquery.PDFQuery(file)

    master_df = pd.DataFrame()

    for i in trange(len(page_list)):
        table = camelot.read_pdf(str(file), flavor="stream", pages=str(page_list[i]))

        # extract the hospital name for that page
        pdf.load(page_list[i] - 1)
        label = pdf.pq('LTTextLineHorizontal:contains("Short Term Acute Care Hospital Discharge Data")')
        left_corner = float(label.attr("x0"))
        bottom_corner = float(label.attr("y0"))
        hospital = pdf.pq(
            'LTTextLineHorizontal:in_bbox("%s, %s, %s, %s")'
            % (left_corner - 40, bottom_corner - 30, left_corner + 500, bottom_corner)
        ).text()

        # extract the dataframe for that page
        df = table[0].df
        df.columns = [i for i in range(df.shape[1])]
        df.rename(columns={df.columns[0]: "variable", df.columns[1]: year}, inplace=True)
        df.replace({"\n": " "}, regex=True, inplace=True)
        df[year] = df[year].str.replace(". ", "0 ", regex=False)

        # Check if current year is in columns, if not skip that hospital
        if sum([sum(df[column].str.contains(year)) for column in df.columns]) < 4:
            print(f"Skipping hospital: {hospital}. It does not have data for year {year}.")
            continue

        # If nan occurs, move info to the left
        df = df.replace("", np.nan)
        if 2 in df.columns:
            df[year][df[year].isnull()] = df[2][df[year].isnull()]
        df = df.replace(np.nan, "")
        keep = [i not in [""] for i in df.variable.values]
        df = df[keep].copy()
        df[year] = [i[0] if len(i) > 0 else "" for i in df[year].str.split(" ")]
        df = df[["variable", year]].copy()
        df[year] = df[year].str.replace(",", "").replace("$", "")
        df = df[df["variable"].isin(keep_columns)].copy()
        df.variable = ["Newborn Count", "Non-Newborn Count", "", "", "", "", "Newborn LOS", "LOS"]
        df = df[df.variable != ""]
        df = df.set_index("variable")
        df.columns = [hospital]

        # Save as column in dataframe
        if master_df.shape[0] == 0:
            master_df = df
        else:
            master_df = pd.merge(master_df, df, left_index=True, right_index=True)

    # Make sure totals match
    master_df.replace(".", "0", inplace=True)
    ss = master_df[[i for i in master_df.columns if i != "Summary Data for All Hospitals"]]
    for col in ["Non-Newborn Count", "Newborn Count"]:
        if ss.loc[col].astype(int).sum() != int(master_df["Summary Data for All Hospitals"].loc[col]):
            print("Problem with totals")

    # Clean and save
    master_df.to_csv(main_dir.joinpath(f"{year}_discharge_final.csv"), index=False)

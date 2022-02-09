import camelot
import pandas as pd
import pdfquery
from data.sheps_data.src.read_ptchar_pdf import main_dir, year
from tqdm import trange


if __name__ == "__main__":
    # specify file and page range here
    file = main_dir.joinpath("ptorg_pt_res_by_hosp_2018.pdf")
    page_range = range(1, 115)
    page_list = list(page_range)

    # query pdf for hospital titles
    pdf = pdfquery.PDFQuery(file)

    master_df = pd.DataFrame()

    for i in trange(len(page_list)):

        table = camelot.read_pdf(str(file), flavor="lattice", pages=str(page_list[i]), process_background=True)

        # extract the hospital name for that page
        pdf.load(page_list[i] - 1)
        label = pdf.pq('LTTextLineHorizontal:contains("Patient County of Residence by Hospital")')
        left_corner = float(label.attr("x0"))
        bottom_corner = float(label.attr("y0"))
        right_corner = float(label.attr("x1"))
        top_corner = float(label.attr("y1"))
        hospital = pdf.pq(
            'LTTextLineHorizontal:in_bbox("%s, %s, %s, %s")' % (left_corner, bottom_corner, right_corner, top_corner)
        ).text()
        hospital = hospital.replace("Patient County of Residence by Hospital - ", "")
        hospital_string = hospital[0 : int(len(hospital) / 2)]
        print(hospital_string)
        # extract the dataframe for that page
        df = table[0].df

        # replace new line symbols with spaces
        df.replace({"\n": " "}, regex=True, inplace=True)

        # set first row to column headers
        df.columns = df.iloc[0]
        df = df.drop(df.index[0])

        df["hospital"] = hospital_string

        mini_df = df[["RESIDENCE", "TOTAL CASES", "hospital"]]

        master_df = pd.concat([master_df, mini_df], axis=0, sort=False)

    final = pd.DataFrame()
    final = master_df.pivot_table(
        index="RESIDENCE", columns="hospital", values="TOTAL CASES", aggfunc=lambda x: " ".join(x)
    )
    final.loc["Actual", :] = final.loc["HOSPITAL TOTAL", :]
    final = final.drop(["HOSPITAL TOTAL"])
    final = final.replace({",": ""}, regex=True).apply(pd.to_numeric, 1)
    final = final.apply(pd.to_numeric)
    final.loc["Calculated", :] = final.iloc[:-1, :].sum(axis=0, skipna=True)
    final.loc["Unreported", :] = final.loc["Actual", :] - final.loc["Calculated", :]
    final.columns = [item.replace("_", " ") for item in final.columns]

    final.to_csv(main_dir.joinpath(f"{year}_master_ptorg_final.csv"))

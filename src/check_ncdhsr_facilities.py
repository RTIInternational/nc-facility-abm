import requests
import pandas as pd

import src.data_input as di


# Header
def main():
    url = "https://info.ncdhhs.gov/dhsr/data/HL-header.txt"
    r = requests.get(url, allow_redirects=True)
    open("data/ncdhsr/HL-header.txt", "wb").write(r.content)

    # Hospital File
    url = "https://info.ncdhhs.gov/dhsr/data/hl.txt"
    r = requests.get(url, allow_redirects=True)
    open("data/ncdhsr/hl.txt", "wb").write(r.content)

    df = pd.read_csv("data/ncdhsr/hl.txt", header=None, sep=",")

    # open the sample file used
    with open("data/ncdhsr/HL-header.txt", "r") as file:
        content = file.readlines()
        headers = [i for i in content[6].split(",") if i != ""]

    df.columns = headers
    df = df[["FID", "FCOUNTY", "FACILITY", "FADDR1", "FCITY"]]
    df.to_csv("data/ncdhsr/facilities.csv", index=False)

    # Check if any new hospitals appear in the DHSR Data
    lt = di.ltachs()
    hospitals = di.hospitals(drop_na=False)
    rehab_centers = [923508, 943092, 70767, 80512, 160338]

    current_facilities = list(hospitals.FID.values) + list(lt.FID.values) + rehab_centers
    needs_review = [i for i in df.FID if i not in current_facilities]
    if len(needs_review) > 0:
        print(f"There are new facilities. Check the facilities.csv file for the following FIDs: {needs_review}.")
    else:
        print("All facilities are in the hospital or LTACH files. No updates are needed.")

    # Check if any hospitals in the SHEPs data are not in the NCDHSR Data
    required_hospitals = di.county_discharges().columns

    for hospital in required_hospitals:
        if hospital not in hospitals["Name"].values:
            print(f"Hospital: {hospital} is not in the NCDHSR data. It may need to be combined with another hospital")


if __name__ == "__main__":
    main()

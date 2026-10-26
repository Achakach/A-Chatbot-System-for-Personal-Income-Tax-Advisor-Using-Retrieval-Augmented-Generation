import os
import zipfile
import requests

# รายการไฟล์ PDF กฎหมายภาษีเงินได้บุคคลธรรมดาและคู่มือทางการ
pdf_sources = {
    # กฎหมายแม่บทและประวัติศาสตร์
    "01_ประมวลรัษฎากร_100ปี.pdf": "https://www.rd.go.th/fileadmin/download/eBook/100revenuecode_121055.pdf",
    "02_พรก_แก้ไขประมวลรัษฎากร_ฉบับ14.pdf": "https://www.rd.go.th/fileadmin/user_upload/kormor/newlaw/prk14.pdf",
    "03_พรบ_แก้ไขประมวลรัษฎากร_ฉบับ53_ธุรกรรมอิเล็กทรอนิกส์.pdf": "https://www.rd.go.th/fileadmin/user_upload/kormor/newlaw/p53.pdf",
    "04_ร่าง_มาตรา3ปัณรส_รายงานธุรกรรมเฉพาะ.pdf": "https://www.rd.go.th/fileadmin/user_upload/kormor/eTax_180860.pdf",
    
    # พระราชกฤษฎีกา และกฎกระทรวง
    "05_พรฎ_ฉบับ750_ยกเว้นภาษีสตาร์ทอัพ_VC.pdf": "https://www.rd.go.th/fileadmin/user_upload/kormor/newlaw/dc750.pdf",
    "06_กฎกระทรวง126_ฉบับประมวลรายการยกเว้น.pdf": "https://www.rd.go.th/fileadmin/images/image_pramoun/file1.2.2.pdf",
    "07_กฎกระทรวง338_ลดหย่อนฝากครรภ์คลอดบุตร.pdf": "https://www.rd.go.th/fileadmin/user_upload/kormor/newlaw/mr338.pdf",
    "08_ประกาศอธิบดี_ยกเว้นภาษีวิสาหกิจชุมชน.pdf": "https://www.rd.go.th/fileadmin/user_upload/kormor/56gr226.pdf",
    
    # คำสั่งกรมสรรพากร (ภาษีต่างประเทศ และ หัก ณ ที่จ่าย)
    "09_คำสั่งป161_2566_เกณฑ์ภาษีเงินได้ต่างประเทศ.pdf": "https://www.rd.go.th/fileadmin/user_upload/kormor/newlaw/dn161A.pdf",
    "10_แนวทางตอบข้อหารือ_ป161_ป162_ภาษีต่างประเทศ.pdf": "https://www.rd.go.th/fileadmin/download/news/question_p161_162.pdf",
    "11_ตารางอัตราหักภาษีณที่จ่าย_ทป4_2528.pdf": "https://www.rd.go.th/fileadmin/download/tab_tp4_101.pdf",
    "12_คำสั่งทป319_2563_แก้ไขทป4.pdf": "https://www.rd.go.th/fileadmin/centralize/news_law_update/2563/01/tp319.pdf",
    "13_คำสั่งทป354_2567_แก้ไขทป4.pdf": "https://www.rd.go.th/fileadmin/user_upload/kormor/newlaw/tp354A.pdf",
    "14_วิเคราะห์มาตรา86_4และ86_6_ใบกำกับภาษี.pdf": "https://www.rd.go.th/fileadmin/images/image_pramoun/mata86_4_6.pdf",
    
    # เอกสารวิชาการ บรรยาย และแนววินิจฉัย
    "15_เอกสารบรรยายPIT_ม40_อ_ปรีชา.pdf": "https://interweb1.rd.go.th/publish/seminar/training/RD17.pdf",
    "16_คู่มือการคำนวณเครดิตภาษีต่างประเทศ_DTA.pdf": "https://www.rd.go.th/fileadmin/user_upload/porphor/GuideTaxFromAbroad.pdf",
    "17_สัมมนาภาษีต่างประเทศ_ThaiESG_eCommerce.pdf": "https://www.rd.go.th/fileadmin/user_upload/lorkhor/newsbanner/2024/2/seminar20_23Feb67.pdf",
    "18_เอกสารสัมมนาหลักแหล่งเงินได้_มาตรา41.pdf": "https://www.rd.go.th/publish/seminar/PIT_doc.pdf",
    "19_เอกสารบรรยายหน้าที่หักณที่จ่าย_อ_จักรภัทร.pdf": "https://interweb1.rd.go.th/publish/seminar/training/RD12.pdf",
    "20_เอกสารบรรยายหน้าที่ผู้จ่ายเงินได้_ทป4.pdf": "https://www.rd.go.th/publish/seminar/180515_WHT_doc.pdf",
    "21_แนววินิจฉัยภาษีบุคคล_ข้อหารือ.pdf": "https://www.rd.go.th/fileadmin/centralize/news_seminar/2562/pdf/25620205_seminar.pdf",
    
    # คู่มือแบบแสดงรายการ (ภ.ง.ด. 90, 91, 94)
    "22_คู่มือกำรกรอกแบบ_ภงด90_ปี2568.pdf": "https://www.rd.go.th/fileadmin/tax_pdf/pit/2568/Ins90_241268.pdf",
    "23_คู่มือกำรกรอกแบบ_ภงด91_ปี2568.pdf": "https://www.rd.go.th/fileadmin/tax_pdf/pit/2568/Ins91_241268.pdf",
    "24_คู่มือกำรกรอกแบบ_ภงด90_ปี2567.pdf": "https://www.rd.go.th/fileadmin/tax_pdf/pit/2567/Ins90_101067.pdf",
    "25_คู่มือกำรกรอกแบบ_ภงด91_ปี2567.pdf": "https://www.rd.go.th/fileadmin/tax_pdf/pit/2567/Ins91_101067.pdf",
    "26_คู่มือภาษีครึ่งปี_ภงด94.pdf": "https://www.rd.go.th/fileadmin/tax_pdf/pit/2569/Ins94_160669.pdf",
    
    # คู่มือเฉพาะกลุ่มธุรกิจและสิทธิประโยชน์
    "27_คู่มือภาษีคริปโทเคอร์เรนซี_ฉบับทางการ.pdf": "https://www.rd.go.th/fileadmin/user_upload/lorkhor/press_release/2022/04/How_to_Crypto__2.pdf",
    "28_คำแนะนำภาษีสินทรัพย์ดิจิทัลและข้อหารือ.pdf": "https://www.rd.go.th/fileadmin/user_upload/lorkhor/information/manual_crypto_310165.pdf",
    "29_คู่มือภาษีธุรกิจค้าทองคำ.pdf": "https://www.rd.go.th/fileadmin/download/morbor_manual_gold.pdf",
    "30_คู่มือภาษีธุรกิจร้านขายยา.pdf": "https://www.rd.go.th/fileadmin/download/drugstore_manual1.pdf",
    "31_คู่มือภาษีวิสาหกิจชุมชน.pdf": "https://www.rd.go.th/fileadmin/user_upload/SMEs/sme_info_100266.pdf",
    "32_คู่มือประกันชีวิตแบบบำนาญ.pdf": "https://www.rd.go.th/fileadmin/user_upload/ebook/pension.pdf",
    "33_คู่มือกฎหมายภาษีกองทุนรวม.pdf": "https://www.rd.go.th/fileadmin/user_upload/morkor/mutual_fund_guide.pdf",
    "34_สรุปรายการค่าลดหย่อนภาษี_อัปเดต2567.pdf": "https://www.rd.go.th/fileadmin/download/tax_deductions_update30072567.pdf"
}

output_dir = "thai_tax_corpus"
os.makedirs(output_dir, exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

print("กำลังเริ่มดาวน์โหลดเอกสาร PDF...")

downloaded_files = []
for filename, url in pdf_sources.items():
    file_path = os.path.join(output_dir, filename)
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
            downloaded_files.append(file_path)
            print(f"✓ สำเร็จ: {filename}")
        else:
            print(f"✗ ล้มเหลว (HTTP {response.status_code}): {filename}")
    except Exception as e:
        print(f"✗ เกิดข้อผิดพลาดกับ {filename}: {e}")

# รวมเป็นไฟล์ ZIP
zip_filename = "thai_pit_rag_corpus.zip"
with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
    for file in downloaded_files:
        zipf.write(file, arcname=os.path.basename(file))

print(f"\nดาวน์โหลดและสร้างไฟล์บีบอัดเรียบร้อยแล้ว: {zip_filename}")
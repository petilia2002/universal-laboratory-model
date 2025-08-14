
import sys
import math 

class Analyte:

    def __init__(self, analyte_id, name):

        self.analyte_id = analyte_id;
        self.name = name;
        self.min = sys.float_info.max;
        self.max = - sys.float_info.max;
        self.add = 0.01;
        self.count = 0;
        self.view_name= None;

    def positive(self, value, gender, useVariance = False) -> float:
        return False;

    def getId(self):
        return self.analyte_id;

    def getName(self):
        return self.name;

    def getViewName(self):
        return self.view_name;

    def setMinimum(self, minValue):
        self.min = minValue;
    
    def setMaximum(self, maxValue):
        self.max = maxValue;

    def getMinimum(self):
        return self.min;
    
    def getMaximum(self):
        return self.max;

    def scale(self, value) -> float:

        add = 0.01 * (self.max - self.min);
        y_max = self.max + add;
        y_min = self.min - add;
  
        if y_max == y_min: return value;

        return 0.9 - 0.8 * (y_max - value) / (y_max - y_min);

    def unscale(self, value) -> float:

        add = 0.01 * (self.max - self.min);
        y_max = self.max + add;
        y_min = self.min - add;
        
        return math.exp(y_max - (0.9 - value) * (y_max - y_min) / 0.8);

    def up(self, value) -> float:
  
        if value is None:
            return None;

        value = math.log(float(value)); 

        self.count = self.count + 1;
        if self.max < value: self.max = value;
        if self.min > value: self.min = value;

        return value;

class Gender(Analyte):
    def __init__(self):
        Analyte.__init__(self, 34, "Gender");
        self.view_name = "gender";
        self.min = 0;
        self.max = 1;

    def up(self, value) -> float:
        self.count = self.count + 1;
        return value;

class Age(Analyte):
    def __init__(self):
        Analyte.__init__(self, 35, "Age");
        self.view_name = "age";

class Hemoglobin(Analyte):
    def __init__(self):
        Analyte.__init__(self, 19, "Hemoglobin");
        self.view_name = "hgb";

class Ferritin(Analyte):

    def __init__(self):
        Analyte.__init__(self, 1, "Ferritin");
        self.view_name = "ferritin";

    def positive(self, value, gender, useVariance = False) -> float:
        if useVariance:
            return 1.0 if value <= 14.925 else 0.0; 
        return 1.0 if value <= 12 else 0.0;

class Uric(Analyte):
    def __init__(self):
       Analyte.__init__(self, 32, "Uric acid");
       self.view_name = "uric";

    def positive(self, value, gender, useVariance = False) -> float:
        if useVariance:
            return  1.0 if ((gender == 1 and value >= 0.4067) or (gender == 0 and value >= 0.322)) else 0.0;
        return 1.0 if ((gender == 1 and value >= 0.48) or (gender == 0 and value >= 0.38)) else 0.0;

class Rbc(Analyte):
    def __init__(self):
        Analyte.__init__(self, 20, "Red Blood Cells");
        self.view_name = "rbc";

class Mcv(Analyte):
    def __init__(self):
        Analyte.__init__(self, 21, "Mean corpuscular volume");
        self.view_name = "mcv";

class Plt(Analyte):
    def __init__(self):
        Analyte.__init__(self, 23, "Platelets");
        self.view_name = "plt";

class Wbc(Analyte):
    def __init__(self):
        Analyte.__init__(self, 22, "White blood cells");
        self.view_name = "wbc";

class Neutrophils(Analyte):
    def __init__(self):
        Analyte.__init__(self, 24, "Neutrophils");
        self.view_name = "neut";

class Lymphocytes(Analyte):
    def __init__(self):
        Analyte.__init__(self, 25, "Lymphocytes");
        self.view_name = "lymph";

class Eosinophils(Analyte):
    def __init__(self):
        Analyte.__init__(self, 26, "Eosinophils");
        self.view_name = "eo";

class Basophils(Analyte):
    def __init__(self):
        Analyte.__init__(self, 27, "Basophils");
        self.view_name = "baso";

class Monocytes(Analyte):
    def __init__(self):
        Analyte.__init__(self, 28, "Monocytes");
        self.view_name = "mono";

class Mid(Analyte):
    def __init__(self):
        Analyte.__init__(self, 36, "Middle-size Cells");
        self.view_name = "mid";

class Gra(Analyte):
    def __init__(self):
        Analyte.__init__(self, 37, "Granulocytes");
        self.view_name = "gra";

class B12(Analyte):
    def __init__(self):
        Analyte.__init__(self, 2, "Vitamin B12");
        self.view_name = "b12";

class Folic(Analyte):
    def __init__(self):
        Analyte.__init__(self, 3, "Folic acid");
        self.view_name = "folic";

class Ast(Analyte):
    def __init__(self):
        Analyte.__init__(self, 4, "Aspartate aminotransferase");
        self.view_name = "ast";
    def positive(self, value, gender, useVariance = False) -> float:        
        return 1.0 if ((value >= 40 and gender == 1) or (value >=32 and gender == 0) ) else 0.0;

class Alt(Analyte):
    def __init__(self):
        Analyte.__init__(self, 5, "Alanine transaminase");
        self.view_name = "alt";
    def positive(self, value, gender, useVariance = False) -> float:        
        return 1.0 if ((value >= 41 and gender == 1) or (value >= 33 and gender == 0) ) else 0.0;

class BilDirect(Analyte):
    def __init__(self):
        Analyte.__init__(self, 6, "Direct bilirubin");
        self.view_name = "bil_direct";

class BilIndirect(Analyte):
    def __init__(self):
        Analyte.__init__(self, 7, "Indirect bilirubin");
        self.view_name = "bil_indirect";

class Biltotal(Analyte):
    def __init__(self):
        Analyte.__init__(self, 8, "Total bilirubin");
        self.view_name = "bil_total";

class Creatinine(Analyte):
    def __init__(self):
        Analyte.__init__(self, 9, "Creatinine");
        self.view_name = "crea";

class Urea(Analyte):
    def __init__(self):
        Analyte.__init__(self, 10, "Urea");
        self.view_name = "urea";

class Pro(Analyte):
    def __init__(self):
        Analyte.__init__(self, 11, "Total protein");
        self.view_name = "pro";

class Ldg(Analyte):
    def __init__(self):
        Analyte.__init__(self, 12, "Lactate dehydrogenase");
        self.view_name = "ldg";

class Albumin(Analyte):
    def __init__(self):
        Analyte.__init__(self, 18, "Albumin");
        self.view_name = "alb";

class Crp(Analyte):
    def __init__(self):
        Analyte.__init__(self, 29, "C-reactive protein");
        self.view_name = "crp";

class Cholesterol(Analyte):
    def __init__(self):
        Analyte.__init__(self, 30, "Cholesterol");
        self.view_name = "chol";

    def positive(self, value, gender, useVariance = False) -> float:
        if useVariance:
            return 1.0 if value >= 4.48 else 0.0; 
        return 1.0 if value >= 5.2 else 0.0;

class Glucose(Analyte):

    def __init__(self):
        Analyte.__init__(self, 31, "Glucose");
        self.view_name = "glu";

    def positive(self, value, gender, useVariance = False) -> float:
        if useVariance:
            return 1.0 if value >= 6.3 else 0.0; 
        return 1.0 if value >= 7.0 else 0.0;

class HbA1c(Analyte):
    def __init__(self):
        Analyte.__init__(self, 41, "Glycated hemoglobin");
        self.view_name = "hba1c";
    def positive(self, value, gender, useVariance = False) -> float:
        if useVariance:
            return 1.0 if value >= 5.52 else 0.0; 
        return 1.0 if value >= 6 else 0.0;
      
class PSA(Analyte):
    def __init__(self):
        Analyte.__init__(self, 42, "Total PSA");
        self.view_name = "psa";
    def positive(self, value, gender, useVariance = False) -> float:
        if useVariance:
            return 1.0 if value >= 2.9 else 0.0; 
        return 1.0 if value >= 4.0 else 0.0;


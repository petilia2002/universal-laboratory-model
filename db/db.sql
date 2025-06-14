/****************** SEQUENCES ********************/

CREATE SEQUENCE GEN_ANALYTES_ID;
CREATE SEQUENCE GEN_ANALYTE_CODES_ID;
CREATE SEQUENCE GEN_EQUIPMENT_ID;
CREATE SEQUENCE GEN_LABS_ID;
CREATE SEQUENCE GEN_LIMITS_ID;
CREATE SEQUENCE GEN_MODELS_ID;
CREATE SEQUENCE GEN_MODEL_PARAMS_IN_ID;
CREATE SEQUENCE GEN_MODEL_PARAMS_OUT_ID;
CREATE SEQUENCE GEN_RESULTS_ID;
CREATE SEQUENCE GEN_SAMPLES_ID;
CREATE SEQUENCE GEN_UNITS_ID;
CREATE SEQUENCE GEN_USERS_ID;

/******************* PROCEDURES ******************/

SET TERM ^ ;
CREATE PROCEDURE ADD_RESULT (
    SAMPLE_ID BIGINT,
    LAB_ID SMALLINT,
    RESULT NUMERIC(8,4),
    LOGDATE TIMESTAMP,
    LIMITS VARCHAR(512),
    STATUS VARCHAR(1),
    ANALYTE_ID SMALLINT,
    UNIT VARCHAR(32),
    EQ VARCHAR(32) )
AS
BEGIN SUSPEND; END^
SET TERM ; ^

SET TERM ^ ;
CREATE PROCEDURE ADD_SAMPLE (
    LAB_ID SMALLINT,
    PID VARCHAR(10),
    SAMPLENO VARCHAR(10),
    AGE SMALLINT,
    GENDER VARCHAR(1),
    COLLECTED TIMESTAMP )
RETURNS (
    ID BIGINT )
AS
BEGIN SUSPEND; END^
SET TERM ; ^

SET TERM ^ ;
CREATE PROCEDURE CLEAN_DATA
AS
BEGIN SUSPEND; END^
SET TERM ; ^

SET TERM ^ ;
CREATE PROCEDURE DEL_SAMPLE (
    ID BIGINT )
AS
BEGIN SUSPEND; END^
SET TERM ; ^

SET TERM ^ ;
CREATE PROCEDURE GET_SAMPLES (
    ID_START BIGINT,
    ID_END BIGINT,
    MODE SMALLINT,
    CNT BIGINT )
RETURNS (
    ID BIGINT,
    LOGDATE DATE,
    AGE SMALLINT,
    GENDER VARCHAR(1),
    HGB NUMERIC(6,2),
    HGB_ST VARCHAR(1),
    RBC NUMERIC(6,2),
    RBC_ST VARCHAR(1),
    MCV NUMERIC(6,2),
    MCV_ST VARCHAR(1),
    WBC NUMERIC(6,2),
    WBC_ST VARCHAR(1),
    PLT NUMERIC(6,2),
    PLT_ST VARCHAR(1),
    NEU NUMERIC(6,2),
    NEU_ST VARCHAR(1),
    LYM NUMERIC(6,2),
    LYM_ST VARCHAR(1),
    EOS NUMERIC(6,2),
    EOS_ST VARCHAR(1),
    BAS NUMERIC(6,2),
    BAS_ST VARCHAR(1),
    MON NUMERIC(6,2),
    MON_ST VARCHAR(1),
    CL1 VARCHAR(16),
    CL1V NUMERIC(5,2),
    CL2 VARCHAR(16),
    CL2V NUMERIC(5,2),
    CL3 VARCHAR(16),
    CL3V NUMERIC(5,2),
    DOC1 VARCHAR(16),
    DOC2 VARCHAR(16),
    DOC3 VARCHAR(16),
    COMMENTS VARCHAR(64),
    CALC SMALLINT,
    VAL SMALLINT )
AS
BEGIN SUSPEND; END^
SET TERM ; ^

SET TERM ^ ;
CREATE PROCEDURE UPDATESTATISTICS
AS
BEGIN SUSPEND; END^
SET TERM ; ^

/******************** TABLES **********************/

CREATE TABLE ANALYTES
(
  ID SMALLINT NOT NULL,
  ANALYTE VARCHAR(32),
  SORTER SMALLINT,
  LOINC VARCHAR(16),
  UNITS VARCHAR(32),
  MIN_VAL NUMERIC(8,4),
  MAX_VAL NUMERIC(8,4),
  VIEW_NAME VARCHAR(32),
  CONSTRAINT INTEG_11 PRIMARY KEY (ID)
);
CREATE TABLE ANALYTE_CODES
(
  ID INTEGER NOT NULL,
  ANALYTE_ID SMALLINT,
  LAB_ID SMALLINT,
  CODE INTEGER,
  COMMENTS VARCHAR(64),
  CONSTRAINT INTEG_13 PRIMARY KEY (ID)
);
CREATE TABLE EQUIPMENT
(
  ID SMALLINT NOT NULL,
  LAB_ID SMALLINT,
  NAME VARCHAR(32),
  CONSTRAINT INTEG_4 PRIMARY KEY (ID)
);
CREATE TABLE LABS
(
  ID SMALLINT NOT NULL,
  NAME VARCHAR(32),
  LASTDATE DATE,
  COMMENTS VARCHAR(126),
  CONSTRAINT INTEG_2 PRIMARY KEY (ID)
);
CREATE TABLE LIMITS
(
  ID BIGINT NOT NULL,
  LIMITS VARCHAR(512),
  CONSTRAINT INTEG_25 PRIMARY KEY (ID),
  CONSTRAINT UNQ_LIMITS_0 UNIQUE (LIMITS)
);
CREATE TABLE MODELS
(
  ID SMALLINT NOT NULL,
  NAME VARCHAR(128),
  COMMENTS VARCHAR(512),
  PATH VARCHAR(256),
  CONSTRAINT INTEG_32 PRIMARY KEY (ID)
);
CREATE TABLE MODEL_PARAMS_IN
(
  ID INTEGER NOT NULL,
  MODEL_ID INTEGER NOT NULL,
  ANALYTE_ID INTEGER NOT NULL,
  SORTER SMALLINT,
  REQUIRED VARCHAR(1),
  CONSTRAINT INTEG_34 PRIMARY KEY (ID)
);
CREATE TABLE MODEL_PARAMS_OUT
(
  ID INTEGER NOT NULL,
  MODEL_ID INTEGER NOT NULL,
  ANALYTE_ID INTEGER NOT NULL,
  SORTER SMALLINT,
  CONSTRAINT INTEG_40 PRIMARY KEY (ID)
);
CREATE TABLE RESULTS
(
  ID BIGINT NOT NULL,
  SAMPLE_ID BIGINT,
  ANALYTE_ID SMALLINT,
  LOGDATE TIMESTAMP,
  RESULT NUMERIC(8,4),
  UNIT_ID SMALLINT,
  STATUS VARCHAR(1),
  EQUIPMENT_ID SMALLINT,
  LIMIT_ID BIGINT,
  CONSTRAINT INTEG_19 PRIMARY KEY (ID)
);
CREATE TABLE SAMPLES
(
  ID BIGINT NOT NULL,
  PID VARCHAR(10),
  LAB_ID SMALLINT,
  SAMPLENO VARCHAR(10) NOT NULL,
  AGE SMALLINT,
  GENDER VARCHAR(1),
  LOGDATE TIMESTAMP,
  CL1 SMALLINT,
  RN1 NUMERIC(5,2),
  CL2 SMALLINT,
  RN2 NUMERIC(5,2),
  CL3 SMALLINT,
  RN3 NUMERIC(5,2),
  RN4 NUMERIC(5,2),
  RN5 NUMERIC(5,2),
  RN6 NUMERIC(5,2),
  RN7 NUMERIC(5,2),
  RN8 NUMERIC(5,2),
  RN9 NUMERIC(5,2),
  CALC SMALLINT,
  VAL SMALLINT,
  DOC1 SMALLINT,
  DOC2 SMALLINT,
  DOC3 SMALLINT,
  COMMENTS VARCHAR(64),
  CONSTRAINT INTEG_7 PRIMARY KEY (ID)
);
CREATE TABLE UNITS
(
  ID SMALLINT NOT NULL,
  UNIT VARCHAR(32),
  CONSTRAINT INTEG_17 PRIMARY KEY (ID)
);
CREATE TABLE USERS
(
  ID SMALLINT NOT NULL,
  EMAIL VARCHAR(20) NOT NULL,
  PASSWD VARCHAR(32) NOT NULL,
  CONSTRAINT INTEG_27 PRIMARY KEY (ID),
  CONSTRAINT INTEG_29 UNIQUE (EMAIL)
);
/********************* VIEWS **********************/

CREATE VIEW LAB (ID, LOGDATE, LAB_ID, SAMPLENO, GENDER, AGE, HGB, RBC, MCV, 
    PLT, WBC, NEUT, LYMPH, EO, BASO, MONO, MID, GRA, FERRITIN, B12, FOLIC, AST, 
    ALT, BIL_DIRECT, BIL_INDIRECT, BIL_TOTAL, CREA, UREA, PRO, LDG, ALB, CRP, 
    CHOL, GLU, URIC, HT, MCH, MCHC, CI, HBA1C, PSA)
AS SELECT 
s.id, s.logdate, s.lab_id, s.sampleno, case s.gender when 'M' then 1 else 0 end as gender, s.age, 
 
(select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 19) as HGB,  
(select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 20) as RBC,
(select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 21) as MCV,
(select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 23) as PLT,
(select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 22) as WBC,
(select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 24) as NEUT,
(select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 25) as LYMPH,
(select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 26) as EO,
(select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 27) as BASO,
(select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 28) as MONO,
(select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 36) as MID,
(select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 37) as GRA,
(select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 1) as FERRITIN,
(select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 2) as B12,
(select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 3) as FOLIC,
(select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 4) as AST,
(select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 5) as ALT,
(select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 6) as BIL_DIRECT,
(select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 7) as BIL_INDIRECT,
(select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 8) as BIL_TOTAL,
(select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 9) as CREA,
(select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 10) as UREA,
(select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 11) as PRO,
(select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 12) as LDG,
(select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 18) as ALB,
(select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 29) as CRP,
(select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 30) as CHOL,
(select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 31) as GLU,
(select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 32) as uric,
(select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 21) * (select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 20) / 10.0,
(select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 19) / (select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 20),
(select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 19) / (select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 21) /
    (select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 20) * 100.0,
3.0 * (select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 19) /
  cast(SUBSTRING( cast((select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 20) * 1000.0 as varchar(32)) from 1 for 3) as double precision),
(select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 41) as Hba1c,
(select r.result from results r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 42) as psa
from samples s;
CREATE VIEW LAB2 (ID, LOGDATE, LAB_ID, SAMPLENO, GENDER, AGE, HGB, RBC, MCV, 
    PLT, WBC, NEUT, LYMPH, EO, BASO, MONO, MID, GRA, FERRITIN, B12, FOLIC, AST, 
    ALT, BIL_DIRECT, BIL_INDIRECT, BIL_TOTAL, CREA, UREA, PRO, LDG, ALB, CRP, 
    CHOL, GLU, URIC, HBA1C, PSA)
AS SELECT 
s.id, s.logdate, s.lab_id, s.sampleno, case s.gender when 'M' then 1 else 0 end as gender, s.age, 
 
(select r.result from res r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 19 and r.result between r.MIN_VAL and r.MAX_VAL) as HGB,  
(select r.result from res r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 20 and r.result between r.MIN_VAL and r.MAX_VAL) as RBC,
(select r.result from res r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 21 and r.result between r.MIN_VAL and r.MAX_VAL) as MCV,
(select r.result from res r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 23 and r.result between r.MIN_VAL and r.MAX_VAL) as PLT,
(select r.result from res r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 22 and r.result between r.MIN_VAL and r.MAX_VAL) as WBC,
(select r.result from res r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 24 and r.result between r.MIN_VAL and r.MAX_VAL) as NEUT,
(select r.result from res r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 25 and r.result between r.MIN_VAL and r.MAX_VAL) as LYMPH,
(select r.result from res r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 26 and r.result between r.MIN_VAL and r.MAX_VAL) as EO,
(select r.result from res r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 27 and r.result between r.MIN_VAL and r.MAX_VAL) as BASO,
(select r.result from res r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 28 and r.result between r.MIN_VAL and r.MAX_VAL) as MONO,
(select r.result from res r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 36 and r.result between r.MIN_VAL and r.MAX_VAL) as MID,
(select r.result from res r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 37 and r.result between r.MIN_VAL and r.MAX_VAL) as GRA,
(select r.result from res r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 1 and r.result between r.MIN_VAL and r.MAX_VAL) as FERRITIN,
(select r.result from res r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 2 and r.result between r.MIN_VAL and r.MAX_VAL) as B12,
(select r.result from res r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 3 and r.result between r.MIN_VAL and r.MAX_VAL) as FOLIC,
(select r.result from res r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 4 and r.result between r.MIN_VAL and r.MAX_VAL) as AST,
(select r.result from res r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 5 and r.result between r.MIN_VAL and r.MAX_VAL) as ALT,
(select r.result from res r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 6 and r.result between r.MIN_VAL and r.MAX_VAL) as BIL_DIRECT,
(select r.result from res r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 7 and r.result between r.MIN_VAL and r.MAX_VAL) as BIL_INDIRECT,
(select r.result from res r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 8 and r.result between r.MIN_VAL and r.MAX_VAL) as BIL_TOTAL,
(select r.result from res r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 9 and r.result between r.MIN_VAL and r.MAX_VAL) as CREA,
(select r.result from res r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 10 and r.result between r.MIN_VAL and r.MAX_VAL) as UREA,
(select r.result from res r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 11 and r.result between r.MIN_VAL and r.MAX_VAL) as PRO,
(select r.result from res r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 12 and r.result between r.MIN_VAL and r.MAX_VAL) as LDG,
(select r.result from res r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 18 and r.result between r.MIN_VAL and r.MAX_VAL) as ALB,
(select r.result from res r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 29 and r.result between r.MIN_VAL and r.MAX_VAL) as CRP,
(select r.result from res r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 30 and r.result between r.MIN_VAL and r.MAX_VAL) as CHOL,
(select r.result from res r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 31 and r.result between r.MIN_VAL and r.MAX_VAL) as GLU,
(select r.result from res r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 32 and r.result between r.MIN_VAL and r.MAX_VAL) as uric,
(select r.result from res r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 41 and r.result between r.MIN_VAL and r.MAX_VAL) as hba1c,
(select r.result from res r where r.SAMPLE_ID = s.id and r.ANALYTE_ID = 42 and r.result between r.MIN_VAL and r.MAX_VAL) as psa
from samples s;
CREATE VIEW RES (ID, SAMPLE_ID, ANALYTE_ID, LOGDATE, RESULT, UNIT_ID, STATUS, 
    EQUIPMENT_ID, LIMIT_ID, MIN_VAL, MAX_VAL)
AS select r.*, a.min_val, a.max_val from results r inner join analytes a on a.id = r.analyte_id;
/******************* EXCEPTIONS *******************/

/******************** TRIGGERS ********************/

SET TERM ^ ;
CREATE TRIGGER TR_ANALYTES_BI FOR ANALYTES ACTIVE
BEFORE INSERT POSITION 0
AS
BEGIN
   if(new.id is null) then 
      new.id = gen_id(GEN_ANALYTES_ID, 1); 
END^
SET TERM ; ^
SET TERM ^ ;
CREATE TRIGGER TR_ANALYTE_CODES_BI FOR ANALYTE_CODES ACTIVE
BEFORE INSERT POSITION 0
AS
BEGIN
    if(new.id is null) then 
       new.id = gen_id(GEN_ANALYTE_CODES_ID, 1);
END^
SET TERM ; ^
SET TERM ^ ;
CREATE TRIGGER TR_EQUIPMENT_BI FOR EQUIPMENT ACTIVE
BEFORE INSERT POSITION 0
AS
BEGIN
   if(new.id is null) then 
      new.id = gen_id(GEN_EQUIPMENT_ID, 1); 
END^
SET TERM ; ^
SET TERM ^ ;
CREATE TRIGGER TR_LABS_BI FOR LABS ACTIVE
BEFORE INSERT POSITION 0
AS
BEGIN
    if(new.id is null) then 
       new.id = gen_id(GEN_LABS_ID, 1); 
END^
SET TERM ; ^
SET TERM ^ ;
CREATE TRIGGER TR_LIMITS_BI FOR LIMITS ACTIVE
BEFORE INSERT POSITION 0
AS
BEGIN
    if(new.id is null) then 
       new.id = gen_id(GEN_LIMITS_ID, 1); 
END^
SET TERM ; ^
SET TERM ^ ;
CREATE TRIGGER TR_MODELS_BI FOR MODELS ACTIVE
BEFORE INSERT POSITION 0
AS
BEGIN
    if(new.id is null) then 
       new.id = gen_id(GEN_MODELS_ID, 1); 
END^
SET TERM ; ^
SET TERM ^ ;
CREATE TRIGGER TR_MODEL_PARAMS_IN_BI FOR MODEL_PARAMS_IN ACTIVE
BEFORE INSERT POSITION 0
AS
BEGIN
    if(new.id is null) then 
       new.id = gen_id(GEN_MODEL_PARAMS_IN_ID, 1);
END^
SET TERM ; ^
SET TERM ^ ;
CREATE TRIGGER TR_MODEL_PARAMS_OUT_BI FOR MODEL_PARAMS_OUT ACTIVE
BEFORE INSERT POSITION 0
AS
BEGIN
    if(new.id is null) then 
       new.id = gen_id(GEN_MODEL_PARAMS_OUT_ID, 1);
END^
SET TERM ; ^
SET TERM ^ ;
CREATE TRIGGER TR_RESULTS_BI FOR RESULTS ACTIVE
BEFORE INSERT POSITION 0
AS
BEGIN
    if(new.id is null) then 
       new.id = gen_id(GEN_RESULTS_ID, 1); 
END^
SET TERM ; ^
SET TERM ^ ;
CREATE TRIGGER TR_SAMPLES_BI FOR SAMPLES ACTIVE
BEFORE INSERT POSITION 0
AS
BEGIN
    if(new.id is null) then 
       new.id = gen_id(GEN_SAMPLES_ID, 1); 
END^
SET TERM ; ^
SET TERM ^ ;
CREATE TRIGGER TR_UNITS_BI FOR UNITS ACTIVE
BEFORE INSERT POSITION 0
AS
BEGIN
    if(new.id is null) then 
       new.id = gen_id(GEN_UNITS_ID, 1); 
END^
SET TERM ; ^
SET TERM ^ ;
CREATE TRIGGER TR_USERS_BI FOR USERS ACTIVE
BEFORE INSERT POSITION 0
AS
BEGIN
    if(new.id is null) then 
       new.id = gen_id(gen_users_id, 1); 
END^
SET TERM ; ^

SET TERM ^ ;
ALTER PROCEDURE ADD_RESULT (
    SAMPLE_ID BIGINT,
    LAB_ID SMALLINT,
    RESULT NUMERIC(8,4),
    LOGDATE TIMESTAMP,
    LIMITS VARCHAR(512),
    STATUS VARCHAR(1),
    ANALYTE_ID SMALLINT,
    UNIT VARCHAR(32),
    EQ VARCHAR(32) )
AS
declare eq_id smallint; 
declare unit_id smallint; 
declare lid bigint; 

BEGIN

   eq_id = null;
   if (eq is not null) then 
   begin       
      select id from equipment where lab_id = :lab_id and name = :eq into eq_id;
      if (eq_id is null) then 
         insert into equipment(lab_id, name) values(:lab_id, :eq) returning id into eq_id;             
   end    
  
   unit_id = null;
   if (unit is not null) then 
   begin       
      select id from units where unit = :unit into unit_id;
      if (unit_id is null) then 
         insert into units(unit) values(:unit) returning id into unit_id;             
   end   
  
   if (not exists(select id from results where sample_id = :sample_id and analyte_id = :analyte_id)) then 
   begin 
   
      lid = null;
      if (limits is not Null) then 
      BEGIN
          select id from limits where limits = :limits into lid;
          if (lid is null) then 
             insert into limits(limits) values(:limits) returning id into lid; 
      end 
   
      insert into results(sample_id, analyte_id, logdate, result, unit_id, limit_id, status, equipment_id) 
         values(:sample_id, :analyte_id, :logdate, :result, :unit_id, :lid, :status, :eq_id); 
   end 
      
END^
SET TERM ; ^


SET TERM ^ ;
ALTER PROCEDURE ADD_SAMPLE (
    LAB_ID SMALLINT,
    PID VARCHAR(10),
    SAMPLENO VARCHAR(10),
    AGE SMALLINT,
    GENDER VARCHAR(1),
    COLLECTED TIMESTAMP )
RETURNS (
    ID BIGINT )
AS
BEGIN

   id = null;
   select id from samples 
       where lab_id = :lab_id and sampleno = :sampleno into id;
       
   if (id is null) then     
   begin 
      insert into samples(lab_id, pid, sampleno, age, gender, logdate) 
          values(:lab_id, :pid, :sampleno, :age, :gender, :collected) returning id into id;
   end 
   
   suspend; 
   
END^
SET TERM ; ^


SET TERM ^ ;
ALTER PROCEDURE CLEAN_DATA
AS
declare id bigint;

declare lymph numeric(5,2);
declare neut numeric(5, 2);
declare eo numeric(5, 2);
declare baso numeric(5, 2);
declare mono numeric(5, 2);
declare gra numeric(5, 2);
declare mid numeric(5, 2);
declare rbc numeric(5,2);
declare hgb numeric(5,2);
declare mcv numeric(5,2);

declare S numeric(5, 2);
declare age int;
declare gender int; 

begin 

for select l.id, l.LYMPH, l.NEUT, l.EO, l.baso, l.MONO, l.mid, l.gra, l.age, l.gender, l.rbc, l.hgb, l.mcv from LAB l  
    into id, lymph, neut, eo, baso, mono, mid, gra, age, gender, rbc, hgb, mcv do 
begin 
 
    if(rbc = 0 or hgb = 0 or mcv = 0) then 
    begin 
        execute procedure del_sample(id);
        continue; 
    end 
    
    if(hgb < 10) then 
    begin 
        execute procedure del_sample(id);
        continue;     
    end 
 
    if (age is null or age > 120 or gender is null) then 
    begin 
        execute procedure del_sample(id);
        continue; 
    end 
 
    if(gra is not null) then 
    begin 
        
        --3DIFF as in Medicalab, Optimalab 
        if( not ( neut is null and eo is null and baso is null and mono is null ) ) then 
            execute procedure del_sample(id);
        else 
        begin 
            S = lymph + mid + gra;
            if (S is null or (S > 104 or S < 96)) then 
                execute procedure del_sample(id);                
        end      
    end 
    else 
    begin 
        if (mid is null) then 
        begin 
            S = lymph + neut + eo + mono + baso;
            if (S is null or (S > 104 or S < 96)) then 
                execute procedure del_sample(id);
        end
        else 
        begin 
            if(eo is not null and baso is not null and mono is not null) then 
            begin 
                S = eo + baso + mono;
                if (S <> mid) then 
                    execute procedure del_sample(id);
            end 
            else if(eo is null and baso is null and mono is null) then 
            begin 
                S =  lymph + neut + mid;
                if (S > 104 or S < 96) then 
                    execute procedure del_sample(id);
            end 
            else
            begin 
                S = coalesce(eo, 0) + coalesce(baso,0) + coalesce(mono,0);
                if (S <> mid) then 
                    execute procedure del_sample(id);
            end             
        end      
    end 
    
end 
end^
SET TERM ; ^


SET TERM ^ ;
ALTER PROCEDURE DEL_SAMPLE (
    ID BIGINT )
AS
begin 

   delete from results r where r.SAMPLE_ID = :id;
   delete from samples where id = :id; 

end^
SET TERM ; ^


SET TERM ^ ;
ALTER PROCEDURE GET_SAMPLES (
    ID_START BIGINT,
    ID_END BIGINT,
    MODE SMALLINT,
    CNT BIGINT )
RETURNS (
    ID BIGINT,
    LOGDATE DATE,
    AGE SMALLINT,
    GENDER VARCHAR(1),
    HGB NUMERIC(6,2),
    HGB_ST VARCHAR(1),
    RBC NUMERIC(6,2),
    RBC_ST VARCHAR(1),
    MCV NUMERIC(6,2),
    MCV_ST VARCHAR(1),
    WBC NUMERIC(6,2),
    WBC_ST VARCHAR(1),
    PLT NUMERIC(6,2),
    PLT_ST VARCHAR(1),
    NEU NUMERIC(6,2),
    NEU_ST VARCHAR(1),
    LYM NUMERIC(6,2),
    LYM_ST VARCHAR(1),
    EOS NUMERIC(6,2),
    EOS_ST VARCHAR(1),
    BAS NUMERIC(6,2),
    BAS_ST VARCHAR(1),
    MON NUMERIC(6,2),
    MON_ST VARCHAR(1),
    CL1 VARCHAR(16),
    CL1V NUMERIC(5,2),
    CL2 VARCHAR(16),
    CL2V NUMERIC(5,2),
    CL3 VARCHAR(16),
    CL3V NUMERIC(5,2),
    DOC1 VARCHAR(16),
    DOC2 VARCHAR(16),
    DOC3 VARCHAR(16),
    COMMENTS VARCHAR(64),
    CALC SMALLINT,
    VAL SMALLINT )
AS
declare analyte smallint; 
  declare res numeric(6, 2);
  declare st varchar(1);
  
  declare n bigint;
  declare s bigint; 
  
BEGIN
  
  n = 0;
  s = 0;
  
  for SELECT s.id, cast(s.logdate as date), s.age, s.gender,
             c1.name, s.rn1, c2.name, s.rn2, c3.name, s.rn3, 
             d1.name, d2.name, d3.name, s.comments, coalesce(s.calc, 0), coalesce(s.val, 0)
      from samples s left join models c1 on c1.id = s.cl1 
                     left join models c2 on c2.id = s.CL2
                     left join models c3 on c3.id = s.cl3
                     left join models d1 on d1.id = s.DOC1
                     left join models d2 on d2.id = s.DOC2
                     left join models d3 on d3.id = s.doc3
      into id, logdate, age, gender, 
            cl1, cl1v, cl2, cl2v, cl3, cl3v, doc1, doc2, doc3, comments, calc, val do
  begin     
  
     if (mode = 1 and calc <> 1) then continue;
     if (mode = 2 and val <> 1) then continue; 
     
     n = n +1;
     
     if (n < id_start) then continue;
     if (n > id_end) then exit;
  
     hgb = null; hgb_st = null;
     rbc = null; rbc_st = null;
     mcv = null; mcv_st = null;
     wbc = null; wbc_st = null;
     plt = null; plt_st = null;
     neu = null; neu_st = null;
     lym = null; lym_st = null;
     eos = null; eos_st = null;
     bas = null; bas_st = null;
     mon = null; mon_st = null;
       
     for select r.analyte_id, r.result, r.status from results r where r.sample_id = :id into analyte, res, st do 
     begin 
         
         if (analyte = 19) then 
         begin             
            hgb = res;
            hgb_st = st;                    
         end 
         
         if (analyte = 20) then 
         begin             
            rbc = res;
            rbc_st = st;                    
         end 
         
         if (analyte = 21) then 
         begin             
            mcv = res;
            mcv_st = st;                    
         end 
         
         if (analyte = 22) then 
         begin             
            wbc = res;
            wbc_st = st;                    
         end 
         
         if (analyte = 23) then 
         begin             
            plt = res;
            plt_st = st;                    
         end 
         
         if (analyte = 24) then 
         begin             
            neu = res;
            neu_st = st;                    
         end 
         
         if (analyte = 25) then 
         begin             
            lym = res;
            lym_st = st;                    
         end 
         
         if (analyte = 26) then 
         begin             
            eos = res;
            eos_st = st;                    
         end 
         
         if (analyte = 27) then 
         begin             
            bas = res;
            bas_st = st;                    
         end 
         
         if (analyte = 28) then 
         begin             
            mon = res;
            mon_st = st;                    
         end 
         
     
     end 

     suspend;
     
     s = s + 1;   
     if (s >= cnt ) then exit;

  end 
  
END^
SET TERM ; ^


SET TERM ^ ;
ALTER PROCEDURE UPDATESTATISTICS
AS
declare indexname varchar(31);
declare sql varchar(512);
begin
    for SELECT RDB$INDEX_NAME FROM RDB$INDICES into :indexname
    do
    begin
        sql = 'set statistics index ' || indexname; 
        execute statement sql;
    end
end^
SET TERM ; ^


comment on column ANALYTES.ANALYTE is 'Это аналит.';
ALTER TABLE ANALYTE_CODES ADD CONSTRAINT INTEG_14
  FOREIGN KEY (ANALYTE_ID) REFERENCES ANALYTES (ID);
ALTER TABLE ANALYTE_CODES ADD CONSTRAINT INTEG_15
  FOREIGN KEY (LAB_ID) REFERENCES LABS (ID);
ALTER TABLE EQUIPMENT ADD CONSTRAINT FK_EQUIPMENT_0
  FOREIGN KEY (LAB_ID) REFERENCES LABS (ID);
ALTER TABLE MODEL_PARAMS_IN ADD CONSTRAINT INTEG_36
  FOREIGN KEY (MODEL_ID) REFERENCES MODELS (ID);
ALTER TABLE MODEL_PARAMS_IN ADD CONSTRAINT INTEG_38
  FOREIGN KEY (ANALYTE_ID) REFERENCES ANALYTES (ID);
ALTER TABLE MODEL_PARAMS_OUT ADD CONSTRAINT INTEG_42
  FOREIGN KEY (MODEL_ID) REFERENCES MODELS (ID);
ALTER TABLE MODEL_PARAMS_OUT ADD CONSTRAINT INTEG_44
  FOREIGN KEY (ANALYTE_ID) REFERENCES ANALYTES (ID);
ALTER TABLE RESULTS ADD CONSTRAINT FK_RESULTS_0
  FOREIGN KEY (LIMIT_ID) REFERENCES LIMITS (ID);
ALTER TABLE RESULTS ADD CONSTRAINT INTEG_20
  FOREIGN KEY (SAMPLE_ID) REFERENCES SAMPLES (ID);
ALTER TABLE RESULTS ADD CONSTRAINT INTEG_21
  FOREIGN KEY (ANALYTE_ID) REFERENCES ANALYTES (ID);
ALTER TABLE RESULTS ADD CONSTRAINT INTEG_22
  FOREIGN KEY (UNIT_ID) REFERENCES UNITS (ID);
ALTER TABLE RESULTS ADD CONSTRAINT INTEG_23
  FOREIGN KEY (EQUIPMENT_ID) REFERENCES EQUIPMENT (ID);
ALTER TABLE SAMPLES ADD CONSTRAINT INTEG_9
  FOREIGN KEY (LAB_ID) REFERENCES LABS (ID);
CREATE INDEX IDX_SAMPLES1 ON SAMPLES (LOGDATE);
CREATE INDEX IDX_UNITS1 ON UNITS (UNIT);

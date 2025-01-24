-- Drop existing tables if they exist
DROP TABLE IF EXISTS temp_dataset;
DROP TABLE IF EXISTS train;
DROP TABLE IF EXISTS test;

-- Create the temporary dataset
CREATE TEMPORARY TABLE temp_dataset AS
SELECT 
    c.formula AS composition,                -- Chemical composition
    fe.delta_e AS target,                    -- Target value (e.g., formation energy)
    fe.stability AS stability,               -- Stability column
    IF(fe.description LIKE '%ICSD%', 1, 0) AS icsd_label, -- Binary ICSD label
    IF(fe.description LIKE '%ICSD%', 1, 0) AS in_icsd,     -- Explicit ICSD flag
    CAST(calc.band_gap AS FLOAT) AS band_gap,               -- Band gap column
    IF(calc.band_gap > 0, 1, 0) AS semic_label, -- Binary Semiconductor label
    RAND() AS rand_num                       -- Random value for splitting
FROM 
    compositions AS c
JOIN 
    formation_energies AS fe ON c.formula = fe.composition_id
JOIN 
    calculations AS calc ON fe.calculation_id = calc.id
WHERE 
    fe.stability <= 0.15;                    -- Stability filter

-- Create train and test sets based on randomized data
CREATE TABLE train AS
SELECT composition, target, stability, icsd_label, in_icsd, band_gap, semic_label
FROM temp_dataset
WHERE rand_num <= 0.9;

CREATE TABLE test AS
SELECT composition, target, stability, icsd_label, in_icsd, band_gap, semic_label
FROM temp_dataset
WHERE rand_num > 0.9;

-- Export train set to CSV
SELECT composition, target, stability, icsd_label, in_icsd, band_gap, semic_label
INTO OUTFILE 'c:\\data\\train.csv'
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
FROM train;

-- Export test set to CSV
SELECT composition, target, stability, icsd_label, in_icsd, band_gap, semic_label
INTO OUTFILE 'c:\\data\\test.csv'
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
FROM test;

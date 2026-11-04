-- Run with a MySQL administrator account, then configure .env.
CREATE DATABASE IF NOT EXISTS mutilagent CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
-- Change this local demo password before using real patient data.
CREATE USER IF NOT EXISTS 'diabetes'@'localhost' IDENTIFIED BY 'diabetes_local';
GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, REFERENCES ON mutilagent.* TO 'diabetes'@'localhost';

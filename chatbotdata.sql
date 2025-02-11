create database chatbotdata;
use chatbotdata;

drop table userdetail;

DESCRIBE userDetail;

CREATE TABLE userDetail (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    phoneNo VARCHAR(20) NOT NULL,
    email VARCHAR(100) NOT NULL,
    query_description VARCHAR(100)
);

select * from userdetail;

DELETE FROM userDetail;


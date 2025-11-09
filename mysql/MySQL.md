# MySQL

# 基本知识

+ 主键：primary key，只能有一个
+ 外键：foreign key，连接不同表格的索引，对应主键
+ 可以设置多个主键组合在一起
+ 一个属性可以同时是主键和外键

# 创建数据库

```sql
create database `database`; 	# 创建数据库，用反引号包裹
show databases;	# 展示数据库
drop database `database`;	# 删除数据库
```

# 创建表格

数据类型：

+ INT：整数
+ DECIMAL（3，2）：小树，小数点后两位
+ VARCHAR（10）：字串
+ BLOB：二进制图片、影片、档案
+ DATE：YYYY-MM-DD
+ TIMESTAMP：YYYY-MM-DD HH：MM：SS

```sql
create database `sql_tutorial`;
show databases;

use `sql_tutorial`;

create table `student`(
	`student_id` INT PRIMARY KEY,
    `name` varchar(20),
    `major` varchar(20)
);

describe `student`;
drop table `student`;

alter table `student` add gpa decimal(3,2);
alter table `student` drop column gpa;
```

# 存储数据

```sql
create table `student`(
	`student_id` INT PRIMARY KEY,
    `name` varchar(20),
    `major` varchar(20)
);

select * from `student`;

insert into `student` values(1, '小白', '数学');
insert into `student` values(2, '小黑', '语文');
insert into `student` values(3, '小蓝', '英语');
insert into `student`(`major`, `student_id`) values('English', 5);
```

# 限制约束

```sql
create table `student`(
	`student_id` INT PRIMARY KEY AUTO_INCREMENT,
    `name` varchar(20) not null,
    `major` varchar(20) unique default '历史'
);

select * from `student`;

insert into `student` values(1, '小白', '数学');
insert into `student` values(2, '小黑', '语文');
insert into `student`(`major`, `student_id`) values('English', 5);


```

# 修改 删除




















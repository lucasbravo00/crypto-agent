-- Incremental migration: adds bingx_order_id to bullets (already included
-- in schema.sql for anyone setting up fresh -- this file is only for an
-- existing project that already ran that once). Run in the SQL Editor.

alter table bullets add column if not exists bingx_order_id text unique;

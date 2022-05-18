format:
	black .
	isort .

clean:
	rm -rf lightning_logs
	rm -rf tb_logs
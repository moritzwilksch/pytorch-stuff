format:
	black .
	isort .

clean:
	rm -r lightning_logs || true
	rm -r tb_logs || true
	rm -r logs/my_model || true
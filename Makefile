install:
\tpython -m pip install --upgrade pip
\tpip install -r requirements.txt

format:
\tblack .
\tisort .

lint:
\truff check .

test:
\tpytest

run:
\tuvicorn app.main:app --reload

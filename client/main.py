# import sys
# sys.path.append('../server')

# from fastapi import FastAPI #import class FastAPI() từ thư viện fastapi
# from check import relation_tagger
# from starlette.responses import HTMLResponse
# app = FastAPI() # gọi constructor và gán vào biến app


# @app.get("/") # giống flask, khai báo phương thức get và url
# async def root(): # do dùng ASGI nên ở đây thêm async, nếu bên thứ 3 không hỗ trợ thì bỏ async đi
#     return relation_tagger()

# @app.get("/html", response_class=HTMLResponse)
# def read_root():
#     with open("../client/interface/index.html", "r") as file:
#         return file.read()

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from feture import named_entity_recognition, relation_tagger, sentiment
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process_text")
async def process_text(request: Request):
    data = await request.json()
    text_input = data.get("text")
    selected_function = data.get("selectedFunction")
    
    # Call the selected function with the text input
    output = call_function(selected_function, text_input)
    
    return {"output": output}


def call_function(selected_function, text_input):
    # Implement the logic to call the selected function here
    if selected_function == "functionA":
        return functionA(text_input)
    elif selected_function == "functionB":
        return functionB(text_input)
    elif selected_function == "functionC":
        return functionC(text_input)
    else:
        return "Invalid function selection"


def functionA(text_input):
    # Function A implementation
    output = named_entity_recognition(text_input)
    return output.__str__()


def functionB(text_input):
    # Function B implementation
    output = relation_tagger(text_input)
    return "Processed by Function B: " + output.__str__()


def functionC(text_input):
    # Function C implementation
    output = sentiment(text_input)
    return "Processed by Function C: " + output.__str__()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Session



# ==========================================
# 1. 数据库配置 (Database Setup)
# ==========================================
# 使用本地 SQLite 数据库文件


SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

# ==========================================
# 2. 创建数据库引擎 (Create Database Engine)
# ==========================================
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)


# ==========================================
# 3. 创建数据库会话 (Create Database Session)
# ==========================================
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ==========================================
# 4. 定义基础模型 (Define Base Model)
# ==========================================
class Base(DeclarativeBase):
    pass


# ==========================================
# 5. 定义数据模型 (Define Data Model)
# ==========================================
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)

# ==========================================
# 6. Pydantic 数据模型 (用于接口请求和响应校验)
# ==========================================
class UserBase(BaseModel):
    name: str
    email: str

    model_config = ConfigDict(
        extra="forbid"  # 禁止额外字段
    )

# 创建时用的模型
class UserCreate(UserBase):
    pass

# 返回给客户端的模型
class UserRead(UserBase):
    id: int

    # 允许 Pydantic 读取 SQLAlchemy 的 ORM 对象属性
    model_config = ConfigDict(
        from_attributes=True  # 允许从 ORM 模型转换
    )

# ==========================================
# 7. FastAPI 应用初始化与依赖项
# ==========================================
# 在数据库中自动创建表结构 (生产环境中通常用 Alembic 做数据迁移)
Base.metadata.create_all(bind=engine)

app = FastAPI()

# 依赖项：获取数据库会话。每次请求分配一个独立会话，请求结束后关闭。
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==========================================
# 8. API 路由定义 (API Route Definitions)
# ==========================================

@app.post("/users/", response_model=UserRead)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = User(name=user.name, email=user.email)
    db.add(db_user) # 添加到会话
    db.commit() # 提交到数据库
    db.refresh(db_user)  # 刷新实例以获取数据库生成的 ID
    return db_user

@app.get("/users/", response_model=List[UserRead])
def read_users(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    users = db.query(User).offset(skip).limit(limit).all()
    return users

@app.get("/users/{user_id}", response_model=UserRead)
def read_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# 删除单个用户
@app.delete("/users/{user_id}", response_model=UserRead)
def delete_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(user)  # 从会话中删除用户
    db.commit()  # 提交删除操作到数据库
    return user
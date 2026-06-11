## 启动测试沙箱容器命令

```bash
docker run -d -p 8080:8080 -p 5900:5900 -p 5901:5901 -p 9222:9222 --name sandbox-dev sandbox-dev
```
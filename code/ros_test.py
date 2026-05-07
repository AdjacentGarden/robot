#!/usr/bin/env python3
"""
测试ROS2控制器子进程运行
演示如何使用subprocess运行controller_cli.py的各种命令
"""

import subprocess
import os
import time

def run_ros2_command(controller_path, action, target_name=None, x=None, y=None, yaw=None):
    """
    运行ROS2控制器命令
    
    Args:
        controller_path: controller_cli.py的完整路径
        action: 命令动作（mapping, navigation, goal-name, stop等）
        target_name: 目标点名称（用于goal-name命令）
        x, y, yaw: 坐标参数（用于goal命令）
    
    Returns:
        bool: 命令执行是否成功
    """
    try:
        # 构建命令列表
        cmd = ["python3", controller_path, action]
        
        # 添加参数
        if target_name:
            cmd.append(target_name)
        elif x is not None and y is not None:
            cmd.extend(["--x", str(x), "--y", str(y)])
            if yaw is not None:
                cmd.extend(["--yaw", str(yaw)])
        
        print(f"🚀 执行命令: {' '.join(cmd)}")
        
        # 运行命令
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=os.path.dirname(controller_path)
        )
        
        # 输出结果
        if result.returncode == 0:
            print(f"✅ 命令执行成功: {action}")
            if result.stdout:
                print(f"输出:\n{result.stdout}")
            return True
        else:
            print(f"❌ 命令执行失败: {action}")
            if result.stderr:
                print(f"错误:\n{result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 命令执行异常: {e}")
        return False

def test_mapping():
    """测试建图功能"""
    print("\n" + "="*50)
    print("测试建图功能")
    print("="*50)
    
    # 修改为您的实际路径
    controller_path = "/home/test/ros2_ws/src/demo/controller_cli.py"
    
    # 检查文件是否存在
    if not os.path.exists(controller_path):
        print(f"❌ 文件不存在: {controller_path}")
        print("请修改controller_path为您的实际路径")
        return False
    
    # 启动建图
    success = run_ros2_command(controller_path, "mapping")
    
    if success:
        print("✅ 建图启动成功，等待5秒...")
        time.sleep(5)
        
        # 停止建图
        print("\n停止建图...")
        run_ros2_command(controller_path, "stop")
    
    return success

def test_navigation():
    """测试导航功能"""
    print("\n" + "="*50)
    print("测试导航功能")
    print("="*50)
    
    controller_path = "/home/test/ros2_ws/src/demo/controller_cli.py"
    
    if not os.path.exists(controller_path):
        print(f"❌ 文件不存在: {controller_path}")
        return False
    
    # 启动导航
    success = run_ros2_command(controller_path, "navigation")
    
    if success:
        print("✅ 导航启动成功，等待3秒...")
        time.sleep(3)
        
        # 导航到目标点
        print("\n导航到目标点...")
        run_ros2_command(controller_path, "goal-name", target_name="home")
        
        time.sleep(3)
        
        # 停止导航
        print("\n停止导航...")
        run_ros2_command(controller_path, "stop")
    
    return success

def test_all_commands():
    """测试所有命令"""
    print("\n" + "="*50)
    print("测试所有ROS2命令")
    print("="*50)
    
    controller_path = "/home/test/ros2_ws/src/demo/controller_cli.py"
    
    if not os.path.exists(controller_path):
        print(f"❌ 文件不存在: {controller_path}")
        return False
    
    # 测试各种命令
    commands = [
        ("status", "获取系统状态"),
        ("points", "列出命名点"),
        ("mapping", "启动建图"),
        ("navigation", "启动导航"),
        ("goal-name", "导航到命名点", "home"),
        ("stop", "停止所有功能")
    ]
    
    for i, command in enumerate(commands):
        print(f"\n[{i+1}/{len(commands)}] 测试: {command[1]}")
        
        if len(command) == 2:
            run_ros2_command(controller_path, command[0])
        else:
            run_ros2_command(controller_path, command[0], target_name=command[2])
        
        # 命令间延迟
        if command[0] not in ["status", "points"]:
            time.sleep(2)

def main():
    """主函数"""
    print("ROS2控制器子进程测试程序")
    print("注意：请确保ROS2环境已正确配置")
    print("注意：请修改controller_path为您的实际路径")
    
    # 测试单个功能
    # test_mapping()
    
    # 测试导航功能
    # test_navigation()
    
    # 测试所有命令
    test_all_commands()
    
    print("\n" + "="*50)
    print("测试完成")
    print("="*50)

if __name__ == "__main__":
    main()
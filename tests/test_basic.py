def test_import():
    """测试基本导入是否正常"""
    import joint_analysis
    assert joint_analysis.__version__ == "0.1.0"

def test_placeholder():
    """一个始终通过的基本测试"""
    assert True
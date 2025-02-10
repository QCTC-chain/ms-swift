import io
from minio import Minio

class Oss(object):
    def __init__(self, access_key_id, access_secret_key, bucket_name, endpoint):
        self.minio_client = Minio(
            endpoint,
            access_key=access_key_id,
            secret_key=access_secret_key,
            secure=False 
        )

        if not self.minio_client.bucket_exists(bucket_name=bucket_name):
            self.create_bucket(bucket_name=bucket_name)

    def create_bucket(self, bucket_name=None):
        """
        创建存储空间
        :param bucket_name:  bucket名称
        :return:
        """
        self.minio_client.make_bucket(bucket_name=bucket_name)
        return True

    def delete_bucket(self, bucket_name=None):
        """
        删除存储空间
        :param bucket_name:  bucket名称
        :return:
        """
        self.minio_client.remove_bucket(bucket_name=bucket_name)
        

    def pub_object(self, bucket_name=None, object_name=None, object_data=None):
        """
        上传文件
            Str
            Bytes
            Unicode
            Stream
        :param bucket_name:  bucket名称
        :param object_name:  不包含Bucket名称组成的Object完整路径
        :param object_data:
        :return:
        """
        if isinstance(object_data, str):
            bytes_data = object_data.encode('utf-8')
        elif isinstance(object_data, bytes):
            bytes_data = object_data
        else:
            raise ValueError('instance object_data not support')

        return self.minio_client.put_object(
            bucket_name, 
            object_name,
            data=io.BytesIO(bytes_data), 
            length=len(bytes_data))

    def put_file(self, bucket_name=None, object_name=None, file_path=None):
        """
        上传文件
            file
        :param bucket_name:  bucket名称
        :param object_name:  不包含Bucket名称组成的Object完整路径
        :param file_path:   文件路径
        :return:
        """
        return self.minio_client.fput_object(
            bucket_name=bucket_name, 
            object_name=object_name, 
            file_path=file_path)

    def delete_objects(self, bucket_name=None, object_name=None):
        """
        批量删除文件
        :param bucket_name:  bucket名称
        :param object_name:  不包含Bucket名称组成的Object完整路径列表
        :return:
        """
        from minio.deleteobjects import DeleteObject
        delete_object_list = map(
            lambda x: DeleteObject(x.object_name),
            self.minio_client.list_objects(bucket_name=bucket_name, recursive=True),
        )
        self.minio_client.remove_objects(bucket_name, delete_object_list)

    def download_object(self, bucket_name=None, object_name=None):
        """
        下载文件到本地
        :param bucket_name:  bucket名称
        :param object_name:  不包含Bucket名称组成的Object完整路径
        :return:
        """
        return self.minio_client.get_object(bucket_name=bucket_name, object_name=object_name)

    def download_file(self, bucket_name=None, object_name=None, save_path=None):
        """
        下载文件到本地
        :param bucket_name:  bucket名称
        :param object_name:  不包含Bucket名称组成的Object完整路径
        :param save_path:  保存路径
        :return:
        """
        self.minio_client.fget_object(
            bucket_name=bucket_name, 
            object_name=object_name, 
            file_path=save_path)
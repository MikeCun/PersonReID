import os
from django.shortcuts import render
from service import PersonReID


# Create your views here.
def upload_file(request):
    """
    Upload the ReID video and the ReID query picture
    :param request: HTTP request
    :return: HTTP Web Page and ReID query result url
    """

    # the url and url_name for query failed
    url = ['/static/fail.jpg']
    url_name = ['fail']

    if request.method == 'POST':

        reid_video_file = request.FILES.getlist('reid_video')[0]
        reid_pic_file = request.FILES.getlist('reid_pic')[0]

        # Handle the uploading action
        handle_uploaded_file(reid_video_file)
        handle_uploaded_file(reid_pic_file)

        # ReID service
        # reid_result = PersonReID(reid_video=os.path.join('reid_query', str(reid_video_file)),
        #                          reid_pic=os.path.join('reid_query', str(reid_pic_file)))
        # url, url_name = reid_result.reid_result()

    context = {'reid_url': zip(url, url_name)}  # API dictionary for showing ReID query result

    return render(request, 'result.html', context)


def index(request):
    """
    Get the Index of the website
    :param request: HTTP request
    :return: index.html
    """

    address = os.getcwd() + '/static/query_result/'
    for file in os.listdir(address):
        os.remove(os.path.join(address + file))

    return render(request, 'index.html')


def handle_uploaded_file(f):
    """
    Handle the uploading action
    :param f: File which will be uploading
    :return: None
    """
    
    # For the first time to create the directory
    if not os.path.exists('./reid_query/'):
        os.makedirs('./reid_query/')

    with open('./reid_query/' + f.name, 'wb+') as destination:
        # Upload the file into 'reid_query' directory
        for chunk in f.chunks():
            destination.write(chunk)

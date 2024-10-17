from setuptools import setup, find_packages

# Read the requirements file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Extract package names
packages = [req.split('==')[0] for req in requirements]

# Read the README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='indoxJudge',
    version='0.0.8',
    license='AGPL-3.0',
    packages=find_packages(),
    package_data={
        'indoxJudge.pipelines.llmEvaluator.graph.assets': ['style.css'],
    },
    include_package_data=True,
    description='Indox Judge',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='osllm',
    author_email='ashkan@nematifamilyfundation.onmicrosoft.com',
    url='https://github.com/osllmai/inDoxJudge',
    keywords=['Evaluation', 'Safety Check', 'LLM Comparison', 'RAG', 'LLM'],
    install_requires=packages,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    python_requires='>=3.9',
)

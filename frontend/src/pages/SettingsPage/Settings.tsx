import React, { useEffect, useState } from 'react';
import FolderPicker from '@/components/FolderPicker/FolderPicker';
import { deleteCache } from '@/services/cacheService';
import { Button } from '@/components/ui/button';
import { restartServer } from '@/utils/serverUtils';
import { isProd } from '../../utils/isProd';
import { FolderSync, Trash2, Server } from 'lucide-react';
import { useLocalStorage } from '@/hooks/LocalStorage';
import LoadingScreen from '@/components/ui/LoadingScreen/LoadingScreen';
import ErrorDialog from '@/components/Album/Error';
import { usePictoMutation } from '@/hooks/useQueryExtensio';
import {
  deleteFolder,
  deleteThumbnails,
  generateThumbnails,
} from '../../../api/api-functions/images.ts';
const Settings: React.FC = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [currentPaths, setCurrentPaths] = useLocalStorage<string[]>(
    'folderPaths',
    [],
  );
  const [autoFolderSetting, setAutoFolder] = useLocalStorage(
    'auto-add-folder',
    'false',
  );
  const [addedFolders, setAddedFolders] = useLocalStorage<string[]>(
    'addedFolders',
    [],
  );
  const [check, setCheck] = useState<boolean>(autoFolderSetting === 'true');
  const [errorDialogContent, setErrorDialogContent] = useState<{
    title: string;
    description: string;
  } | null>(null);
  const { mutate: generateThumbnailsAPI, isPending: isGeneratingThumbnails } =
    usePictoMutation({
      mutationFn: generateThumbnails,
      autoInvalidateTags: ['ai-tagging-images', 'ai'],
    });

  const { mutate: deleteFolderAITagging } = usePictoMutation({
    mutationFn: deleteFolder,
    autoInvalidateTags: ['ai-tagging-images', 'ai'],
  });

  const { mutate: deleteThumbnail, isPending: isDeletingThumbnails } =
    usePictoMutation({
      mutationFn: deleteThumbnails,
    });

  useEffect(() => {
    setCheck(autoFolderSetting === 'true');
  }, [autoFolderSetting]);

  const handleFolderPathChange = async (newPaths: string[]) => {
    //Error if newPaths contains a path that is already in currentPaths
    const duplicatePaths = newPaths.filter((path) =>
      currentPaths.includes(path),
    );
    if (duplicatePaths.length > 0) {
      showErrorDialog(
        'Duplicate Paths',
        new Error(
          `The following paths are already selected: ${duplicatePaths.join(', ')}`,
        ),
      );
      return;
    }
    generateThumbnailsAPI([...currentPaths, ...newPaths]);
    setCurrentPaths([...currentPaths, ...newPaths]);
    await deleteCache();
  };
  const handleDeleteCache = async () => {
    try {
      const result = await deleteCache();
      if (result) {
        console.log('Cache deleted');
      }
    } catch (error) {
      console.error('Error deleting cache:', error);
    }
  };

  const handleRemoveFolderPath = async (pathToRemove: string) => {
    try {
      const updatedPaths = currentPaths.filter((path) => path !== pathToRemove);
      setCurrentPaths(updatedPaths);
      setAddedFolders(addedFolders.filter((path) => path !== pathToRemove));
      deleteThumbnail(pathToRemove);
      deleteFolderAITagging(pathToRemove);
      await deleteCache();
      console.log(`Removed folder path: ${pathToRemove}`);
    } catch (error) {
      console.error('Error removing folder path:', error);
    }
  };

  if (isGeneratingThumbnails || isDeletingThumbnails || isLoading) {
    return (
      <div className="flex h-full w-full items-center justify-center">
        <LoadingScreen variant="fullscreen" />
      </div>
    );
  }

  const showErrorDialog = (title: string, err: unknown) => {
    setErrorDialogContent({
      title,
      description:
        err instanceof Error ? err.message : 'An unknown error occurred',
    });
  };

  return (
    <div className="mx-auto flex-1 px-4 pt-1">
      <div className="bg-theme-light rounded-2xl space-y-6 border border-white/10 p-6 shadow backdrop-blur-md backdrop-saturate-150 dark:border-white/5 dark:bg-white/5">
        <div>
          <h2 className="text-theme-dark dark:text-theme-light mb-2 text-lg font-medium">
            Current Folder Paths
          </h2>
          {currentPaths.length > 0 ? (
            currentPaths.map((path, index) => (
              <div
                key={index}
                className="text-theme-dark dark:text-theme-light mt-2 flex items-center justify-between rounded-md border bg-gray-100 px-4 py-2 backdrop-blur-sm dark:border-white/5 dark:bg-white/5"
              >
                <span className="truncate">{path}</span>
                <Button
                  onClick={() => handleRemoveFolderPath(path)}
                  variant="outline"
                  className="h-10 w-10 border-gray-500 p-2 hover:bg-red-500 hover:text-white"
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            ))
          ) : (
            <div className="text-theme-dark dark:text-theme-light mt-4 flex items-center justify-between rounded-md border bg-gray-100 px-4 py-2 backdrop-blur-sm dark:border-white/5 dark:bg-white/5">
              No folder paths selected
            </div>
          )}
        </div>
        <div className="w-40 space-y-4">
          <FolderPicker
            setFolderPaths={handleFolderPathChange}
            className="h-10 w-full"
          />
          <Button
            onClick={handleDeleteCache}
            variant="outline"
            className="h-10 w-full border-gray-500 hover:bg-accent dark:hover:bg-white/10"
          >
            <FolderSync className="text-gray-5 mr-2 h-5 w-5 dark:text-gray-50" />
            Refresh Cache
          </Button>
          {isProd() && (
            <Button
              onClick={() => restartServer(setIsLoading)}
              variant="outline"
              className="h-10 w-full border-gray-500 hover:bg-accent dark:hover:bg-white/10"
            >
              <Server className="text-gray-5 mr-2 h-5 w-5 dark:text-gray-50" />
              Restart Server
            </Button>
          )}
        </div>
        <div>
          <label className="inline-flex cursor-pointer items-center gap-2">
            <input
              type="checkbox"
              value=""
              className="peer sr-only"
              checked={check}
              onClick={() => {
                setCheck(!check);
                setAutoFolder(check ? 'false' : 'true');
              }}
            />
            <span className="ms-3 text-sm font-medium text-gray-900 dark:text-gray-300">
              Auto Sync Desktop Folders
            </span>
            <div className="rounded-full after:rounded-full peer relative h-6 w-11 bg-gray-200 after:absolute after:start-[2px] after:top-0.5 after:h-5 after:w-5 after:border after:border-gray-300 after:bg-white after:transition-all after:content-[''] peer-checked:bg-blue-600 peer-checked:after:translate-x-full peer-checked:after:border-white peer-focus:ring-4 peer-focus:ring-blue-300 dark:border-gray-600 dark:bg-gray-700 dark:peer-checked:bg-blue-600 dark:peer-focus:ring-blue-800 rtl:peer-checked:after:-translate-x-full"></div>
          </label>
          <p className="ml-3 mt-1 text-xs text-yellow-500">
            WARNING: It may impact performance, restart for changes to take
            effect.
          </p>
        </div>
      </div>
      <ErrorDialog
        content={errorDialogContent}
        onClose={() => setErrorDialogContent(null)}
      />
    </div>
  );
};

export default Settings;
